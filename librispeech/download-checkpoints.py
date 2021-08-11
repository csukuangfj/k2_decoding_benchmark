#!/usr/bin/env python3
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


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["pretrainer"].collect_files()

    hparams["pretrainer"].load_collected(device="cpu")


if __name__ == "__main__":
    main()
