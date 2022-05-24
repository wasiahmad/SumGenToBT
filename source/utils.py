import os
import sys
import logging
import itertools
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from .language_pair_dataset import LanguagePairDataset

from evaluation.utils import (
    EXT,
    TOFILL,
    return_script_not_found,
    run_java_program,
    run_cpp_program,
    run_python_program
)

logger = logging.getLogger(__name__)


def submit_functions(
        lang_processor,
        functions_list,
        id,
        ref,
        lang,
        outfolder,
        script_folder,
        roberta_mode=False,
):
    results_list = []
    i = id.rstrip()
    for try_id, f_fill in enumerate(functions_list):
        f = f_fill.rstrip()
        script_model_path = os.path.join(script_folder, f"{lang}/{i}.{EXT[lang]}")
        if os.path.exists(script_model_path):
            script_model = open(script_model_path, "r", encoding="utf-8").read()
            try:
                f_name = lang_processor.get_function_name(f)
                f = f.replace(f_name, "f_filled")
            except:
                results_list.append(("error", "Could not replace function name"))
            if f_fill == ref:
                results_list.append(("success", "identical to gold"))
                return results_list, i
            f = (
                lang_processor.detokenize_code(f)
                if not roberta_mode
                else f.replace("#NEWLINE", "\n")
            )
            script = script_model.replace(TOFILL[lang], f)
            if lang == "python":
                script = f"import numpy as np \nimport math\nfrom math import *\nimport collections\n" \
                         f"from collections import *\nimport heapq\nimport itertools\nimport random\n" \
                         f"import sys\n\n{script}"
            ############# This is done by Wasi to handle JavaFX issues #############
            if lang == 'java':
                script = script.replace('import javafx.util.Pair;', '// import javafx.util.Pair;')
            ########################################################################
            script_path = f"{outfolder}/{i}.{EXT[lang]}"
            open(script_path, "w", encoding="utf-8").write(script)
            run_pg = globals()[f"run_{lang}_program"]
            result, _ = run_pg(script_path, i)
            if result[0] == "success":
                results_list.append(result)
                return results_list, i
            results_list.append(result)
        else:
            return [return_script_not_found()], i
    return results_list, i


def load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        prepend_bos_src=None,
        append_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)
    elif append_bos_src is not None:
        logger.info(f"appending src bos: {append_bos_src}")
        src_dataset = AppendTokenDataset(src_dataset, append_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
    )
