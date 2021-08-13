#!/usr/bin/env python3
import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["pretrainer"].collect_files()

    hparams["pretrainer"].load_collected(device="cpu")


if __name__ == "__main__":
    main()
