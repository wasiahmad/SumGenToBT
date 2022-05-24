# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import argparse
import subprocess

from collections import OrderedDict
from evaluation.bleu import compute_bleu
from evaluation.CodeBLEU.calc_code_bleu import compute_codebleu
from lang_processors.java_processor import JavaProcessor
from lang_processors.python_processor import PythonProcessor
from lang_processors.cpp_processor import CppProcessor
from pathlib import Path

root_directory = Path(__file__).parents[1].joinpath("third_party")
LANG_PROCESSORS = {
    'java': JavaProcessor(root_folder=root_directory),
    'python': PythonProcessor(root_folder=root_directory),
    'cpp': CppProcessor(root_folder=root_directory)
}

BLEU_SCRIPT_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "multi-bleu.perl"
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def ignore_prefix(code, prefixes):
    for prefix in prefixes:
        if code.startswith(prefix):
            return code[len(prefix):].strip()
    return code


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--references', required=True, help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', nargs='+', type=str, required=True,
                        help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument('--lang', type=str, required=True, help='name of the programming language',
                        choices=['java', 'python', 'cpp'])
    parser.add_argument('--detokenize', action='store_true',
                        help="detokenize both predictions and reference code.")
    args = parser.parse_args()

    refs = [x.strip() for x in open(args.references, 'r', encoding='utf-8').readlines()]
    predictions = [[
        x.strip() for x in open(pred_file, 'r', encoding='utf-8').readlines()
    ] for pred_file in args.predictions]

    length_match = [len(refs) == len(p) for p in predictions]
    assert all(length_match)

    scores = OrderedDict([
        ("EM", []),
        ("BLEU", []),
        ("ngram_match_score", []),
        ("weighted_ngram_match_score", []),
        ("syntax_match_score", []),
        ("dataflow_match_score", []),
        ("CodeBLEU", []),
    ])
    for preds in predictions:
        EM = 0.0
        translations = []
        references = []
        split_translations = []
        split_references = []
        for pred, ref in zip(preds, refs):
            if args.lang == 'java':
                pred = ignore_prefix(pred, ['public'])
                ref = ignore_prefix(ref, ['public'])
            elif args.lang == 'python':
                pass
            elif args.lang == 'cpp':
                pred = ignore_prefix(pred, ['private :', 'public :'])
                ref = ignore_prefix(ref, ['private :', 'public :'])

            if args.detokenize:
                pred = LANG_PROCESSORS[args.lang].detokenize_code(pred)
                ref = LANG_PROCESSORS[args.lang].detokenize_code(ref)

            if pred == ref:
                EM += 1

            translations.append(pred)
            references.append([ref])
            split_translations.append(pred.split())
            split_references.append([ref.split()])

        bleu_score, _, _, _, _, _ = compute_bleu(split_references, split_translations, 4, True)
        scores['EM'].append((EM / len(refs)) * 100.0)
        scores['BLEU'].append(bleu_score * 100.0)

        if args.lang in ['java', 'python']:
            code_bleu_score, (ngram_match_score, weighted_ngram_match_score, syntax_match_score,
                              dataflow_match_score) = compute_codebleu(translations, references, args.lang)
            scores['ngram_match_score'].append(ngram_match_score * 100.0)
            scores['weighted_ngram_match_score'].append(weighted_ngram_match_score * 100.0)
            scores['syntax_match_score'].append(syntax_match_score * 100.0)
            scores['dataflow_match_score'].append(dataflow_match_score * 100.0)
            scores['CodeBLEU'].append(code_bleu_score * 100.0)

    scores['MOSES BLEU'] = eval_moses_bleu(args.references, args.predictions[0])

    for k, v in scores.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        elif isinstance(v, list) and v:
            print(f"{k}: {max(v):.2f}")


if __name__ == "__main__":
    main()
