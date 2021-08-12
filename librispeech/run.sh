#!/usr/bin/env bash

set -x

lang_dir=data/lang_bpe
dl_dir=./download
lm_dir=data/lm

. path.sh


mkdir -p $lang_dir
mkdir -p $dl_dir/lm
mkdir -p $lm_dir

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ ! -f $dl_dir/lm/.done ]; then
  log "Downloading LM"
  ./local/download_lm.py --out-dir=$dl_dir/lm
  touch $dl_dir/lm/.done
fi

if [ ! -f $lang_dir/bpe.model ]; then
  log "Downloading checkpoints"
  python3 ./download-checkpoints.py ./params.yaml
  cp exp/tokenizer.ckpt $lang_dir/bpe.model
fi

if [ ! -f $lang_dir/words.txt ]; then
  ( echo "<eps>"
    cat $dl_dir/lm/librispeech-vocab.txt
    echo "#0"
    echo "<UNK>"
    echo "<s>"
    echo "</s>"
  ) | awk '{print $1, NR}' > $lang_dir/words.txt
fi

if [ ! -f data/lang_bpe/L_disambig.pt ]; then
  ./local/prepare_lang_bpe.py --lang-dir $lang_dir
fi

# We assume you have install kaldilm, if not, please install
# it using: pip install kaldilm

if [ ! -f $lm_dir/G_3_gram.fst.txt ]; then
  # It is used in building HLG
  python3 -m kaldilm \
    --read-symbol-table="$lang_dir/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    $dl_dir/lm/3-gram.pruned.1e-7.arpa > $lm_dir/G_3_gram.fst.txt
fi

if [ ! -f $lm_dir/G_4_gram.fst.txt ]; then
  # It is used for LM rescoring
  python3 -m kaldilm \
    --read-symbol-table="$lang_dir/words.txt" \
    --disambig-symbol='#0' \
    --max-order=4 \
    $dl_dir/lm/4-gram.arpa > $lm_dir/G_4_gram.fst.txt
fi

./local/compile_hlg.py --lang-dir $lang_dir

python3 ./main.py ./params.yaml
