import re
import os
import json
import argparse
import subprocess

from tqdm import tqdm
from lang_processors.java_processor import JavaProcessor
from lang_processors.python_processor import PythonProcessor
from lang_processors.cpp_processor import CppProcessor

DOCSTRING_REGEX_TOKENIZER = re.compile(
    r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")


def tokenize_docstring(docstring):
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstring) if t is not None and len(t) > 0]


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def prepare(args):
    javaProcessor = JavaProcessor(root_folder=args.root_folder)
    pythonProcessor = PythonProcessor(root_folder=args.root_folder)
    cppProcessor = CppProcessor(root_folder=args.root_folder)

    for split in ['train', 'valid', 'test']:
        filename = '{}/{}.jsonl'.format(args.lang, split)
        if not os.path.exists(filename):
            continue

        src_writer = open(
            os.path.join(
                args.out_dir, '{}.{}-en_XX.{}'.format(split, args.lang, args.lang)
            ), 'w', encoding='utf-8'
        )
        tgt_writer = open(
            os.path.join(
                args.out_dir, '{}.{}-en_XX.en_XX'.format(split, args.lang)
            ), 'w', encoding='utf-8'
        )
        with open(filename) as f:
            for line in tqdm(
                    f, total=count_file_lines(filename), desc="{}-{}".format(args.lang, split)
            ):
                ex = json.loads(line.strip())
                if 'docstring_tokens' in ex:
                    docstring = ' '.join(ex['docstring_tokens'])
                elif 'summary' in ex:
                    docstring = ' '.join(tokenize_docstring(ex['summary']))
                docstring = re.sub("[\n\r\t ]+", " ", docstring).strip()
                code = ex['code'] if 'code' in ex else ex['function']
                if len(code) == 0 or len(docstring) == 0:
                    continue
                try:
                    if args.lang == 'java':
                        code_tokens = javaProcessor.tokenize_code(code)
                        if len(code_tokens) == 0:
                            continue
                        if args.keep_standalone_only:
                            if "static" not in code_tokens[0: code_tokens.index("{")]:
                                continue
                    elif args.lang == 'python':
                        code_tokens = pythonProcessor.tokenize_code(code)
                        if len(code_tokens) == 0:
                            continue
                        if args.keep_standalone_only:
                            if code_tokens[code_tokens.index("(") + 1] == "self":
                                continue
                    else:
                        code_tokens = cppProcessor.tokenize_code(code)
                        if len(code_tokens) == 0:
                            continue
                        if args.keep_standalone_only:
                            pass

                    code = ' '.join(code_tokens)
                    src_writer.write(code + '\n')
                    tgt_writer.write(docstring + '\n')
                except:
                    continue

            src_writer.close()
            tgt_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help='Language name'
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help='Keep standalone function only'
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
    args = parser.parse_args()
    prepare(args)
