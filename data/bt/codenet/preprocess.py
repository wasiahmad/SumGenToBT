import os
import glob
import json
import argparse

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from lang_processors.java_processor import JavaProcessor
from lang_processors.python_processor import PythonProcessor


def initialize_processors(root_folder):
    global jprocessor, pyprocessor, cpprocessor
    jprocessor = JavaProcessor(root_folder=root_folder)
    pyprocessor = PythonProcessor(root_folder=root_folder)


def process_example(ex, keep_standalone_only):
    global jprocessor, pyprocessor
    processor = jprocessor if args.lang == 'java' else pyprocessor
    try:
        tokens = processor.tokenize_code(ex['code'])
        if len(tokens) == 0:
            return None
        functions_standalone, functions_class = processor.extract_functions(tokens)
        if keep_standalone_only:
            return functions_standalone
        else:
            return functions_standalone + functions_class
    except:
        return None


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
        fn = partial(process_example, keep_standalone_only=args.keep_standalone_only)
        with tqdm(total=len(data), desc="{}-{}".format(args.lang, filename)) as pbar:
            for i, out in enumerate(pool.map(fn, data, 1000)):
                pbar.update()
                if out is not None:
                    results.extend(out)

        for tokenized_code in results:
            try:
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
