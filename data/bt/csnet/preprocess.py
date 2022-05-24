import os
import glob
import json
import pickle
import argparse
import subprocess

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from lang_processors.java_processor import JavaProcessor
from lang_processors.python_processor import PythonProcessor
from lang_processors.cpp_processor import CppProcessor


def initialize_processors(root_folder):
    global jprocessor, pyprocessor, cpprocessor
    jprocessor = JavaProcessor(root_folder=root_folder)
    pyprocessor = PythonProcessor(root_folder=root_folder)
    cpprocessor = CppProcessor(root_folder=root_folder)


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def process_bimodal_instance(ex, keep_standalone_only):
    global jprocessor, pyprocessor, cpprocessor
    # docstring = ' '.join(ex['docstring_tokens'])
    # docstring = re.sub("[\n\r\t ]+", " ", docstring).strip()
    # if len(docstring) == 0:
    #     return None
    try:
        if args.lang == 'java':
            tokens = jprocessor.tokenize_code(ex['code'])
            if len(tokens) == 0:
                return None
            if keep_standalone_only:
                if "static" not in tokens[0: tokens.index("{")]:
                    return None
        elif args.lang == 'python':
            tokens = pyprocessor.tokenize_code(ex['code'])
            if len(tokens) == 0:
                return None
            if keep_standalone_only:
                if tokens[tokens.index("(") + 1] == "self":
                    return None
        elif args.lang == 'cpp':
            tokens = cpprocessor.tokenize_code(ex['code'])
            functions_standalone, functions_class = cpprocessor.extract_functions(tokens)
            assert len(functions_standalone) + len(functions_class) == 1
            if keep_standalone_only and len(functions_standalone) == 0:
                return None
            function = functions_standalone[0] if functions_standalone else functions_class[0]
            tokens = cpprocessor.tokenize_code(function)
            if len(tokens) == 0:
                return None
        else:
            return None
        tokenized_code = ' '.join(tokens)
    except:
        return None

    return tokenized_code


def process_unimodal_instance(ex, keep_standalone_only):
    global jprocessor, pyprocessor, cpprocessor
    tokenized_code = ''
    if len(ex['docstring_tokens']) == 0 and 'function' in ex:
        # unimodal data / only function
        try:
            if args.lang == 'java':
                tokens = jprocessor.tokenize_code(ex['function'])
                if len(tokens) == 0:
                    return None
                if keep_standalone_only:
                    if "static" not in tokens[0: tokens.index("{")]:
                        return None
            elif args.lang == 'python':
                tokens = pyprocessor.tokenize_code(ex['function'])
                if len(tokens) == 0:
                    return None
                if keep_standalone_only:
                    if tokens[tokens.index("(") + 1] == "self":
                        return None
            elif args.lang == 'cpp':
                # TODO: can we directly identify if function is standalone?
                tokens = cpprocessor.tokenize_code(ex['code'])
                functions_standalone, functions_class = cpprocessor.extract_functions(tokens)
                assert len(functions_standalone) + len(functions_class) == 1
                if keep_standalone_only and len(functions_standalone) == 0:
                    return None
                function = functions_standalone[0] if functions_standalone else functions_class[0]
                tokens = cpprocessor.tokenize_code(function)
                if len(tokens) == 0:
                    return None
            else:
                return None
            tokenized_code = ' '.join(tokens)
        except:
            return None

    return tokenized_code


def prepare(args):
    initialize_processors(args.root_folder)
    pool = Pool(min(cpu_count(), args.workers))
    src_dir = os.path.join(args.source_dir, args.lang)
    pl_writer = open(
        '{}/{}.functions.tok'.format(args.target_pl_dir, args.split), 'w', encoding='utf-8'
    )
    for file in glob.glob("{}/{}_{}_*.jsonl".format(src_dir, args.lang, args.split)):
        filename, _ = os.path.splitext(os.path.basename(file))
        with open(file) as f:
            data = [json.loads(line.strip()) for line in f]

        results = []
        fn = partial(process_bimodal_instance, keep_standalone_only=args.keep_standalone_only)
        with tqdm(total=len(data), desc="{}-{}".format(args.lang, filename)) as pbar:
            for i, out in enumerate(pool.map(fn, data, 1000)):
                pbar.update()
                if out is not None:
                    results.append(out)

        for tokenized_code in results:
            try:
                pl_writer.write(tokenized_code + '\n')
            except:
                pass

    if args.split == 'train':
        num_unimodal_ex = 0
        filename = "{}/{}_dedupe_definitions_v2.pkl".format(src_dir, args.lang)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)

                results = []
                fn = partial(process_unimodal_instance, keep_standalone_only=args.keep_standalone_only)
                with tqdm(total=len(data), desc="unimodal-data") as pbar:
                    for i, out in enumerate(pool.map(fn, data, 1000)):
                        pbar.update()
                        if out is not None:
                            results.append(out)

                for tokenized_code in results:
                    num_unimodal_ex += 1
                    try:
                        # write may through UnicodeEncodeError
                        pl_writer.write(tokenized_code + '\n')
                    except:
                        pass

    pl_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", help='Language name',
    )
    parser.add_argument(
        "--source_dir", help='Source directory',
    )
    parser.add_argument(
        "--target_pl_dir", help="Output directory to save tokenized functions",
    )
    parser.add_argument(
        "--target_nl_dir", help="Output directory to save tokenized docstrings",
    )
    parser.add_argument(
        "--split", type=str, default='train', help='Dataset split',
    )
    parser.add_argument(
        "--keep_standalone_only",
        action='store_true',
        help='Keep standalone function only'
    )
    parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help='Root folder where tree-sitter repos are stored'
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    prepare(args)
