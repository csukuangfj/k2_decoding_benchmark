#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

"""

This script takes as input `lang_dir`, which should contain::

    - lang_dir/bpe.model,
    - lang_dir/words.txt

and generates the following files in the directory `lang_dir`:

    - lexicon.txt
    - lexicon_disambig.txt
    - L.pt
    - L_disambig.pt
    - tokens.txt
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import k2
import sentencepiece as spm
import torch

Lexicon = List[Tuple[str, List[str]]]


def write_lexicon(filename: str, lexicon: List[Tuple[str, List[str]]]) -> None:
    """Write a lexicon to a file.

    Args:
      filename:
        Path to the lexicon file to be generated.
      lexicon:
        It can be the return value of :func:`read_lexicon`.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word, tokens in lexicon:
            f.write(f"{word} {' '.join(tokens)}\n")


def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:
    """Write a symbol to ID mapping to a file.

    Note:
      No need to implement `read_mapping` as it can be done
      through :func:`k2.SymbolTable.from_file`.

    Args:
      filename:
        Filename to save the mapping.
      sym2id:
        A dict mapping symbols to IDs.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")


def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    """It adds pseudo-token disambiguation symbols #1, #2 and so on
    at the ends of tokens to ensure that all pronunciations are different,
    and that none is a prefix of another.

    See also add_lex_disambig.pl from kaldi.

    Args:
      lexicon:
        It is returned by :func:`read_lexicon`.
    Returns:
      Return a tuple with two elements:

        - The output lexicon with disambiguation symbols
        - The ID of the max disambiguation symbol that appears
          in the lexicon
    """

    # (1) Work out the count of each token-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, tokens in lexicon:
        count[" ".join(tokens)] += 1

    # (2) For each left sub-sequence of each token-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, tokens in lexicon:
        tokens = tokens.copy()
        tokens.pop()
        while tokens:
            issubseq[" ".join(tokens)] = 1
            tokens.pop()

    # (3) For each entry in the lexicon:
    # if the token sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same token-seq
    # has already been assigned a disambig symbol.
    ans = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, tokens in lexicon:
        tokenseq = " ".join(tokens)
        assert tokenseq != ""
        if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
            ans.append((word, tokens))
            continue

        cur_disambig = last_used_disambig_symbol_of[tokenseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[tokenseq] = cur_disambig
        tokenseq += f" #{cur_disambig}"
        ans.append((word, tokenseq.split()))
    return ans, max_disambig


def add_self_loops(
    arcs: List[List[Any]], disambig_token: int, disambig_word: int
) -> List[List[Any]]:
    """Adds self-loops to states of an FST to propagate disambiguation symbols
    through it. They are added on each state with non-epsilon output symbols
    on at least one arc out of the state.

    See also fstaddselfloops.pl from Kaldi. One difference is that
    Kaldi uses OpenFst style FSTs and it has multiple final states.
    This function uses k2 style FSTs and it does not need to add self-loops
    to the final state.

    The input label of a self-loop is `disambig_token`, while the output
    label is `disambig_word`.

    Args:
      arcs:
        A list-of-list. The sublist contains
        `[src_state, dest_state, label, aux_label, score]`
      disambig_token:
        It is the token ID of the symbol `#0`.
      disambig_word:
        It is the word ID of the symbol `#0`.

    Return:
      Return new `arcs` containing self-loops.
    """
    states_needs_self_loops = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for s in states_needs_self_loops:
        ans.append([s, s, disambig_token, disambig_word, 0])

    return arcs + ans


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []

    assert word2id["<eps>"] == 0

    eps = 0

    for word, pieces in lexicon:
        assert len(pieces) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        pieces = [token2id[i] for i in pieces]

        for i in range(len(pieces) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, pieces[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last piece of this word
        i = len(pieces) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, pieces[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs,
            disambig_token=disambig_token,
            disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def generate_lexicon(
    model_file: str, words: List[str]
) -> Tuple[Lexicon, Dict[str, int]]:
    """Generate a lexicon from a BPE model.

    Args:
      model_file:
        Path to a sentencepiece model.
      words:
        A list of strings representing words.
    Returns:
      Return a tuple with two elements:
        - A dict whose keys are words and values are the corresponding
          word pieces.
        - A dict representing the token symbol, mapping from tokens to IDs.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    words_pieces: List[List[str]] = sp.encode(words, out_type=str)

    lexicon = []
    for word, pieces in zip(words, words_pieces):
        lexicon.append((word, pieces))

    # The OOV word is <UNK>
    lexicon.append(("<UNK>", [sp.id_to_piece(sp.unk_id())]))

    token2id: Dict[str, int] = dict()
    for i in range(sp.vocab_size()):
        token2id[sp.id_to_piece(i)] = i

    return lexicon, token2id


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        It should contain the bpe.model and words.txt
        """,
    )

    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    model_file = lang_dir / "bpe.model"

    word_sym_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

    words = word_sym_table.symbols

    excluded = ["<eps>", "<UNK>", "#0", "<s>", "</s>"]
    for w in excluded:
        if w in words:
            words.remove(w)

    lexicon, token_sym_table = generate_lexicon(model_file, words)

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    next_token_id = max(token_sym_table.values()) + 1
    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in token_sym_table
        token_sym_table[disambig] = next_token_id
        next_token_id += 1

    word_sym_table.add("#0")
    word_sym_table.add("<s>")
    word_sym_table.add("</s>")

    write_mapping(lang_dir / "tokens.txt", token_sym_table)

    write_lexicon(lang_dir / "lexicon.txt", lexicon)
    write_lexicon(lang_dir / "lexicon_disambig.txt", lexicon_disambig)

    L = lexicon_to_fst_no_sil(
        lexicon,
        token2id=token_sym_table,
        word2id=word_sym_table,
    )

    L_disambig = lexicon_to_fst_no_sil(
        lexicon_disambig,
        token2id=token_sym_table,
        word2id=word_sym_table,
        need_self_loops=True,
    )
    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")


if __name__ == "__main__":
    main()
