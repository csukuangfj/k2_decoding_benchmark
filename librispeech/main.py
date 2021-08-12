#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from local.common import get_datasets, prepare_data_csv
from local.decode import (
    get_lattice,
    nbest_decoding,
    one_best_decoding,
    rescore_with_attention_decoder,
    rescore_with_n_best_list,
    rescore_with_whole_lattice,
)
from local.lexicon import Lexicon
from local.utils import (
    setup_logger,
    AttributeDict,
    get_texts,
    store_transcripts,
    write_error_stats,
)
from speechbrain.dataio.dataio import length_to_mask


def get_params() -> AttributeDict:
    """Read params from params.yaml."""
    params_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)
        params = AttributeDict(params)

    if "select_n_sentences" not in params:
        params.select_n_sentences = None
    elif len(params.select_n_sentences) == 0:
        params.select_n_sentences = None

    params.device = torch.device(params.device)
    params.out_dir = Path(params.out_dir)
    params.lm_dir = Path(params.lm_dir)

    return params


def compute_features(params, batch):
    device = params.device
    batch = batch.to(device)
    wavs, wav_lens = batch.sig

    feats = params.compute_features(wavs)
    feats = params.normalize(feats, wav_lens)
    src = params.CNN(feats)

    # src is [N, T, C1, C2]
    src = src.reshape(src.shape[0], src.shape[1], -1)
    # Now src is [N, T, C]

    return src, wav_lens


def run_transformer_encoder(params: AttributeDict, batch):
    model = params.Transformer

    src, wav_lens = compute_features(params, batch)

    # Note: We don't use model.encode() here
    # since it ignores src_key_padding_mask
    # during the test time
    return model.encode(src, wav_lens), wav_lens

    abs_len = torch.round(wav_lens * src.shape[1])
    src_key_padding_mask = (1 - length_to_mask(abs_len)).bool()
    src_key_padding_mask = None

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
    return encoder_out, src_key_padding_mask, wav_lens


def decode_one_batch(
    params: AttributeDict,
    HLG: k2.Fsa,
    batch,
    lexicon: Lexicon,
    sos_id: int,
    eos_id: int,
    G: Optional[k2.Fsa] = None,
) -> Dict[str, List[List[int]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if no rescoring is used, the key is the string `no_rescore`.
               If LM rescoring is used, the key is the string `lm_scale_xxx`,
               where `xxx` is the value of `lm_scale`. An example key is
               `lm_scale_0.7`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding without LM rescoring.
        - params.method is "nbest", it uses nbest decoding without LM rescoring.
        - params.method is "nbest-rescoring", it uses nbest LM rescoring.
        - params.method is "whole-lattice-rescoring", it uses whole lattice LM
          rescoring.

      model:
        The neural model.
      HLG:
        The decoding graph.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      lexicon:
        It contains word symbol table.
      sos_id:
        The token ID of the SOS.
      eos_id:
        The token ID of the EOS.
      G:
        An LM. It is not None when params.method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    memory, wav_lens = run_transformer_encoder(params, batch)

    abs_len = torch.round(wav_lens * memory.shape[1])
    src_key_padding_mask = (1 - length_to_mask(abs_len)).bool()

    logits = params.ctc_lin(memory)
    nnet_output = params.log_softmax(logits)
    # nnet_output is [N, T, C]

    supervision_segments = torch.zeros(nnet_output.shape[0], 3)
    supervision_segments[:, 0] = torch.arange(nnet_output.shape[0])

    abs_len = torch.round(wav_lens * nnet_output.shape[1])
    supervision_segments[:, 2] = abs_len

    supervision_segments = supervision_segments.to(torch.int32)

    lattice = get_lattice(
        nnet_output=nnet_output,
        HLG=HLG,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=1,
    )

    if params.method in ["1best", "nbest"]:
        if params.method == "1best":
            best_path = one_best_decoding(
                lattice=lattice, use_double_scores=params.use_double_scores
            )
            key = "no_rescore"
        else:
            best_path = nbest_decoding(
                lattice=lattice,
                num_paths=params.num_paths,
                use_double_scores=params.use_double_scores,
            )
            key = f"no_rescore-{params.num_paths}"

        hyps = get_texts(best_path)
        hyps = [[lexicon.word_table[i] for i in ids] for ids in hyps]
        return {key: hyps}

    assert params.method in [
        "nbest-rescoring",
        "whole-lattice-rescoring",
        "attention-decoder",
    ]

    lm_scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lm_scale_list += [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    lm_scale_list += [1.4, 1.5, 1.6]

    if params.method == "nbest-rescoring":
        best_path_dict = rescore_with_n_best_list(
            lattice=lattice,
            G=G,
            num_paths=params.num_paths,
            lm_scale_list=lm_scale_list,
        )
    elif params.method == "whole-lattice-rescoring":
        best_path_dict = rescore_with_whole_lattice(
            lattice=lattice, G_with_epsilon_loops=G, lm_scale_list=lm_scale_list
        )
    elif params.method == "attention-decoder":
        # lattice uses a 3-gram Lm. We rescore it with a 4-gram LM.
        rescored_lattice = rescore_with_whole_lattice(
            lattice=lattice, G_with_epsilon_loops=G, lm_scale_list=None
        )

        best_path_dict = rescore_with_attention_decoder(
            lattice=rescored_lattice,
            num_paths=params.num_paths,
            model=None,  # TODO: fix me
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask,
            sos_id=sos_id,
            eos_id=eos_id,
        )
    else:
        assert False, f"Unsupported decoding method: {params.method}"

    ans = dict()
    for lm_scale_str, best_path in best_path_dict.items():
        hyps = get_texts(best_path)
        hyps = [[lexicon.word_table[i] for i in ids] for ids in hyps]
        ans[lm_scale_str] = hyps
    return ans


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
):
    if params.method == "attention-decoder":
        # Set it to False since there are too many logs.
        enable_log = False
    else:
        enable_log = True
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.out_dir / f"recogs-{test_set_name}-{key}.txt"
        store_transcripts(filename=recog_path, texts=results)
        if enable_log:
            logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.out_dir / f"errs-{test_set_name}-{key}.txt"
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=enable_log
            )
            test_set_wers[key] = wer

        if enable_log:
            logging.info(
                "Wrote detailed error stats to {}".format(errs_filename)
            )

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.out_dir / f"wer-summary-{test_set_name}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    HLG: k2.Fsa,
    lexicon: Lexicon,
    sos_id: int,
    eos_id: int,
    G: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[List[int], List[int]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      HLG:
        The decoding graph.
      lexicon:
        It contains word symbol table.
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
      G:
        An LM. It is not None when params.method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    results = []

    tot_num = len(dl)

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch.wrd

        hyps_dict = decode_one_batch(
            params=params,
            HLG=HLG,
            batch=batch,
            lexicon=lexicon,
            G=G,
            sos_id=sos_id,
            eos_id=eos_id,
        )

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for hyp_words, ref_text in zip(hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        if batch_idx % 20 == 0:
            logging.info(
                f"batch {batch_idx}, batches processed until now is "
                f"{batch_idx}/{tot_num} "
                f"({float(batch_idx)/tot_num*100:.6f}%)"
            )
    return results


@torch.no_grad()
def main():
    params = get_params()

    prepare_data_csv(params)

    params.pretrainer.collect_files()

    params.pretrainer.load_collected(device="cpu")

    datasets = get_datasets(params)

    device = params.device

    for key, value in params.items():
        if isinstance(value, torch.nn.Module):
            value.to(device)
            value.eval()

    params.ctc_topo = k2.ctc_topo(
        params.tokenizer.vocab_size() - 1, modified=True, device=device
    )

    lexicon = Lexicon(params.lang_dir)

    HLG = k2.Fsa.from_dict(torch.load(f"{params.lang_dir}/HLG.pt"))
    HLG = HLG.to(device)
    assert HLG.requires_grad is False

    if not hasattr(HLG, "lm_scores"):
        HLG.lm_scores = HLG.scores.clone()

    if params.method in (
        "nbest-rescoring",
        "whole-lattice-rescoring",
        "attention-decoder",
    ):
        if not (params.lm_dir / "G_4_gram.pt").is_file():
            logging.info("Loading G_4_gram.fst.txt")
            logging.warning("It may take 8 minutes (will be cached).")
            with open(params.lm_dir / "G_4_gram.fst.txt") as f:
                first_word_disambig_id = lexicon.word_table["#0"]

                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                # G.aux_labels is not needed in later computations, so
                # remove it here.
                del G.aux_labels
                # CAUTION: The following line is crucial.
                # Arcs entering the back-off state have label equal to #0.
                # We have to change it to 0 here.
                G.labels[G.labels >= first_word_disambig_id] = 0
                G = k2.Fsa.from_fsas([G]).to(device)
                G = k2.arc_sort(G)
                torch.save(G.as_dict(), params.lm_dir / "G_4_gram.pt")
        else:
            logging.info("Loading pre-compiled G_4_gram.pt")
            d = torch.load(params.lm_dir / "G_4_gram.pt")
            G = k2.Fsa.from_dict(d).to(device)

        if params.method in ["whole-lattice-rescoring", "attention-decoder"]:
            # Add epsilon self-loops to G as we will compose
            # it with the whole lattice later
            G = k2.add_epsilon_self_loops(G)
            G = k2.arc_sort(G)
            G = G.to(device)

        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
    else:
        G = None

    for name, dataset in datasets.items():
        logging.info(f"Decode {name} started")
        dataloader = sb.dataio.dataloader.make_dataloader(
            dataset, batch_size=params.batch_size
        )

        results_dict = decode_dataset(
            dl=dataloader,
            params=params,
            HLG=HLG,
            lexicon=lexicon,
            G=G,
            sos_id=params.bos_index,
            eos_id=params.eos_index,
        )

        save_results(
            params=params, test_set_name=name, results_dict=results_dict
        )


if __name__ == "__main__":
    Path("exp").mkdir(exist_ok=True)
    log_file = "exp/log-decode"
    setup_logger(log_file)

    main()
