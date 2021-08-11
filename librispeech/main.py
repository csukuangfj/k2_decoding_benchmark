#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

import logging
import sys
from pathlib import Path
from typing import Dict

import k2
import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.decoders.k2 import ctc_decoding
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import length_to_mask

from prepare import prepare_librispeech


class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_params() -> AttributeDict:
    """Read params from params.yaml."""
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        hparams = AttributeDict(hparams)

    if "select_n_sentences" not in hparams:
        hparams.select_n_sentences = None
    elif len(hparams.select_n_sentences) == 0:
        hparams.select_n_sentences = None

    hparams.device = torch.device(hparams.device)

    return hparams


def prepare_data_csv(hparams: AttributeDict) -> None:
    """Prepare the librispeech test datasets.

    The generated files are saved in `hparams.out_dir`.
    """
    out_dir = Path(hparams.out_dir)
    if (out_dir / ".done").is_file():
        logging.info("Skipping data preparation")
        return
    prepare_librispeech(
        data_folder=hparams.dataset_dir,
        save_folder=hparams.out_dir,
        te_splits=hparams.test_splits,
        select_n_sentences=hparams.select_n_sentences,
        skip_prep=False,
    )
    (out_dir / ".done").touch()


def get_datasets(hparams) -> Dict[str, sb.dataio.dataset.DynamicItemDataset]:
    """
    Return a dict with keys being test-clean and test-other
    and with values being sb.dataio.dataset.DynamicItemDataset.
    """
    test_datasets = {}
    for csv_file in hparams.test_csv:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(sort_key="duration")

    datasets = list(test_datasets.values())

    tokenizer = hparams.tokenizer

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return test_datasets


def compute_features(hparams, batch):
    device = hparams.device
    batch = batch.to(device)
    wavs, wav_lens = batch.sig

    feats = hparams.compute_features(wavs)
    feats = hparams.normalize(feats, wav_lens)
    src = hparams.CNN(feats)

    # src is [N, T, C1, C2]
    src = src.reshape(src.shape[0], src.shape[1], -1)
    # Now src is [N, T, C]

    return src, wav_lens


def run_transformer_encoder(hparams: AttributeDict, batch):
    model = hparams.Transformer

    src, wav_lens = compute_features(hparams, batch)

    # Note: We don't use model.encode() here
    # since it ignores src_key_padding_mask
    # during the test time

    abs_len = torch.round(wav_lens * src.shape[1])
    src_key_padding_mask = (1 - length_to_mask(abs_len)).bool()

    src = model.custom_src_module(src)
    if model.attention_type == "RelPosMHAXL":
        pos_embs_source = model.positional_encoding(src)
    elif model.positional_encoding_type == "fixed_abs_sine":
        src = src + model.positional_encoding(src)
        pos_embs_source = None

    encoder_out, _ = model.encoder(
        src=src,
        src_key_padding_mask=src_key_padding_mask,
        pos_embs=pos_embs_source,
    )
    return encoder_out, src_key_padding_mask


def run_ctc_decoding(hparams, batch):
    encoder_out, _ = run_transformer_encoder(hparams, batch)

    logits = hparams.ctc_lin(encoder_out)
    p_ctc = hparams.log_softmax(logits)
    hyps, _ = ctc_decoding(p_ctc.detach(), hparams.ctc_topo)
    return hyps


def main():
    hparams = get_params()

    prepare_data_csv(hparams)

    hparams.pretrainer.collect_files()

    hparams.pretrainer.load_collected(device="cpu")

    datasets = get_datasets(hparams)

    for key, value in hparams.items():
        if isinstance(value, torch.nn.Module):
            value.to(hparams.device)

    hparams.ctc_topo = k2.ctc_topo(
        hparams.tokenizer.vocab_size() - 1, modified=True, device=hparams.device
    )

    for name, dataset in datasets.items():
        wer_metric = hparams.error_rate_computer()

        dataloader = sb.dataio.dataloader.make_dataloader(
            dataset, batch_size=hparams.batch_size
        )
        assert isinstance(dataloader, DataLoader)

        num_batches = len(dataloader)

        logging.info(f"Decode {name} started")
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0:
                logging.info(f"Processing {batch_idx}/{num_batches}")
            hyps = run_ctc_decoding(hparams, batch)

            predicted_words = [
                hparams.tokenizer.decode(utt_seq).split(" ") for utt_seq in hyps
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            wer_metric.append(batch.id, predicted_words, target_words)

        logging.info(f"Decode {name} Done")
        logging.info(f"{name}: {wer_metric.summarize()}")
        filename = Path(hparams.out_dir) / f"wer-{name}.txt"
        with open(filename, "w") as f:
            wer_metric.write_stats(f)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
