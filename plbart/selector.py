import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from prettytable import PrettyTable
from evaluation.pl_eval import BLEU_SCRIPT_PATH

from evaluation.utils import (
    bool_flag,
    eval_function_output
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO, stream=sys.stdout
)
logger = logging.getLogger(__name__)

EVAL_SCRIPT_FOLDER = {
    "test": "../data/transcoder_evaluation_gfg",
    "valid": "../data/transcoder_evaluation_gfg"
}


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + "0")
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + " %s < %s"
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith("BLEU"):
        return float(result[7: result.index(",")])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1


def main(params):
    best_validation_score = -1
    best_validation_ckpt_name = ''
    filepattern = f'{params.source_lang}_{params.target_lang}_checkpoint_*.output'
    results = {}
    for output_file in Path(params.hyp_dir).glob(filepattern):
        shutil.rmtree(params.outfolder, ignore_errors=True)
        Path(params.outfolder).mkdir(parents=True, exist_ok=True)
        func_run_stats, func_run_out = eval_function_output(
            params.ref_path,
            [output_file],
            params.id_path,
            params.target_lang,
            params.outfolder,
            EVAL_SCRIPT_FOLDER[params.split],
            params.retry_mismatching_types,
            roberta_mode=False
        )
        ckpt_filename = output_file.stem.replace(
            f'{params.source_lang}_{params.target_lang}_', ''
        )
        log_string = "%s_%s-%s_%s" % (params.split, params.source_lang, params.target_lang, ckpt_filename)
        logger.info("Computation res %s : %s" % (log_string, json.dumps(func_run_stats)))
        comp_acc = func_run_stats['success'] / (
            func_run_stats['total_evaluated'] if func_run_stats['total_evaluated'] else 1)
        logger.info("%s = %f" % (log_string, comp_acc))

        if ckpt_filename not in results:
            results[ckpt_filename] = dict()
        results[ckpt_filename]["Comp-Acc"] = round(comp_acc * 100, 2)
        results[ckpt_filename]["BLEU"] = round(
            eval_moses_bleu(params.ref_path, output_file), 2
        )

        if comp_acc > best_validation_score:
            best_validation_score = comp_acc
            best_validation_ckpt_name = ckpt_filename

    logger.info("best validation comp-acc = %f" % (best_validation_score))
    logger.info("best validation checkpoint name = %s" % (best_validation_ckpt_name))

    table = PrettyTable()
    table.field_names = ["Checkpoint-Name", "BLEU", "Comp-Acc"]
    table.align["Checkpoint-Name"] = "l"
    table.align["BLEU"] = "c"
    table.align["Comp-Acc"] = "c"

    num_steps = [int(k.split("_")[-1]) for k in results]
    num_steps.sort()
    for s in num_steps:
        for k, v in results.items():
            update_num = int(k.split("_")[-1])
            if s == update_num:
                table.add_row([k, v["BLEU"], v["Comp-Acc"]])
                break
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, required=True, help="Path to references")
    parser.add_argument("--id_path", type=str, required=True, help="Path to identities")
    parser.add_argument("--hyp_dir", type=str, required=True, help="Path to hypotheses directory")
    parser.add_argument("--split", type=str, default='test', help="Dataset split")
    parser.add_argument("--outfolder", type=str, required=True, help="Output directory")
    parser.add_argument("--source_lang", type=str, required=True, help="Source language")
    parser.add_argument("--target_lang", type=str, required=True, help="Target language")
    parser.add_argument("--retry_mismatching_types", type=bool_flag, default=False,
                        help="Retry with wrapper at eval time when the types do not match")

    params = parser.parse_args()
    main(params)
