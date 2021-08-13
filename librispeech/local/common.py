import logging
from pathlib import Path
from typing import Dict

import speechbrain as sb
import torch
from prepare import prepare_librispeech


class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def prepare_data_csv(params: AttributeDict) -> None:
    """Prepare the librispeech test datasets.

    The generated files are saved in `params.out_dir`.
    """
    if (params.out_dir / ".done").is_file():
        logging.info("Skipping data preparation")
        return
    prepare_librispeech(
        data_folder=params.dataset_dir,
        save_folder=params.out_dir,
        te_splits=params.test_splits,
        select_n_sentences=params.select_n_sentences,
        skip_prep=False,
    )
    (params.out_dir / ".done").touch()


def get_datasets(params) -> Dict[str, sb.dataio.dataset.DynamicItemDataset]:
    """
    Return a dict with keys being test-clean and test-other
    and with values being sb.dataio.dataset.DynamicItemDataset.
    """
    test_datasets = {}
    for csv_file in params.test_csv:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file
        )
        #  test_datasets[name] = test_datasets[name].filtered_sorted(
        #      sort_key="duration"
        #  )

    datasets = list(test_datasets.values())

    tokenizer = params.tokenizer

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
        tokens_bos = torch.LongTensor([params["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [params["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return test_datasets
