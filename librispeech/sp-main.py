#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
import logging
import os
import sys
from pathlib import Path

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from local.common import get_datasets, prepare_data_csv
from local.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    write_error_stats,
)
from speechbrain.pretrained import EncoderDecoderASR


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

    if "dataset_dir" in os.environ:
        params.dataset_dir = os.environ["dataset_dir"]
        logging.info(f"dataset_dir: {params.dataset_dir}")

    params.device = torch.device(params.device)
    params.out_dir = Path(params.out_dir)

    return params


def main():
    params = get_params()

    prepare_data_csv(params)

    params.pretrainer.collect_files()

    params.pretrainer.load_collected(device="cpu")

    run_opts = {"device": params.device}
    model = EncoderDecoderASR(
        modules=params.modules, hparams=params, run_opts=run_opts
    )

    datasets = get_datasets(params)

    for name, dataset in datasets.items():
        logging.info(f"Decode {name} started")
        dataloader = sb.dataio.dataloader.make_dataloader(
            dataset, batch_size=params.batch_size
        )
        results = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                logging.info(f"Processing {batch_idx}/{len(dataloader)}")
            texts = batch.wrd

            wavs, wav_lens = batch.sig
            hyps, _ = model.transcribe_batch(wavs, wav_lens)

            hyps = [h.split() for h in hyps]
            refs = [t.split() for t in texts]

            for ref, hyp in zip(refs, hyps):
                results.append((ref, hyp))

        recog_path = params.out_dir / f"recogs-{name}.txt"
        store_transcripts(filename=recog_path, texts=results)

        errs_filename = params.out_dir / f"errs-{name}.txt"

        with open(errs_filename, "w") as f:
            write_error_stats(f, name, results)


if __name__ == "__main__":
    Path("exp-sp").mkdir(exist_ok=True)
    log_file = "exp-sp/log-decode"
    setup_logger(log_file)

    main()
