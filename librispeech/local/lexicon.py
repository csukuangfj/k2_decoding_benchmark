import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import k2
import torch


class Lexicon(object):
    def __init__(
        self,
        lang_dir: Path,
        disambig_pattern: str = re.compile(r"^#\d+$"),
    ):
        """
        Args:
          lang_dir:
            Path to the lang director. It is expected to contain the following
            files:
                - tokens.txt
                - words.txt
            The above files are produced by the script `prepare.sh`. You
            should have run that before running the training code.
          disambig_pattern:
            It contains the pattern for disambiguation symbols.
        """
        lang_dir = Path(lang_dir)
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

        self.disambig_pattern = disambig_pattern

    @property
    def tokens(self) -> List[int]:
        """Return a list of token IDs excluding those from
        disambiguation symbols.

        Caution:
          0 is not a token ID so it is excluded from the return value.
        """
        symbols = self.token_table.symbols
        ans = []
        for s in symbols:
            if not self.disambig_pattern.match(s):
                ans.append(self.token_table[s])
        if 0 in ans:
            ans.remove(0)
        ans.sort()
        return ans
