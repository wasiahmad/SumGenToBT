import os
import glob
import json
import csv
import argparse

from tqdm import tqdm
from pathlib import Path


def read_meta_data(filename):
    file = open(filename)
    csvreader = csv.reader(file)
    # submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy
    header = next(csvreader)
    status_idx = header.index('status')
    submission_id_idx = header.index('submission_id')
    data = {}
    for row in csvreader:
        if len(row) == len(header):
            data[row[submission_id_idx]] = row[status_idx].lower()
    file.close()
    return data


def load_ignore_problems(source_dir):
    identical_problem_clusters_dir = os.path.join(source_dir, 'derived', 'duplicates')
    ignore_problems = set()
    with open(os.path.join(identical_problem_clusters_dir, 'identical_problem_clusters')) as f:
        for line in f:
            ignore_problems.update(line.strip().split(',')[1:])

    return ignore_problems


def load_outliers(source_dir, lang):
    outliers = set()
    folder_name = 'Java' if lang == 'java' else 'Python'
    target_dir = os.path.join(source_dir, 'derived', 'duplicates', folder_name)
    with open(os.path.join(target_dir, '{}_accepted_outliers.txt'.format(folder_name))) as f:
        for line in f:
            # /Volume1/AI4CODE/CodeNet/data/p03017/Java/s398852335.java
            # problem_id = Path(line).parts[-3]
            submission_id = os.path.splitext(os.path.basename(line.strip()))[0]
            outliers.add(submission_id)
    with open(os.path.join(target_dir, '{}_rejected_outliers.txt'.format(folder_name))) as f:
        for line in f:
            submission_id = os.path.splitext(os.path.basename(line.strip()))[0]
            outliers.add(submission_id)

    return outliers


def load_singletons_from_clusters(source_dir, lang):
    singletons = set()
    folder_name = 'Java' if lang == 'java' else 'Python'
    target_dir = os.path.join(source_dir, 'derived', 'duplicates', folder_name)
    clusters = []
    with open(os.path.join(target_dir, 'Project_CodeNet-{}.clusters'.format(folder_name))) as f:
        for line in f:
            if len(line.strip()) == 0:
                if clusters:
                    singletons.add(clusters[0])
                    clusters = []
                continue
            # /Volume1/AI4CODE/CodeNet/data/p03017/Java/s398852335.java
            submission_id = os.path.splitext(os.path.basename(line.strip()))[0]
            clusters.append(submission_id)

    if clusters:
        singletons.add(clusters[0])
    return singletons


def prepare(args):
    ignore_problems = load_ignore_problems(args.source_dir)
    outliers = load_outliers(args.source_dir, args.lang)
    singletons = load_singletons_from_clusters(args.source_dir, args.lang)

    data_dir = os.path.join(args.source_dir, 'data')
    meta_data_dir = os.path.join(args.source_dir, 'metadata')
    data = []

    problem_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for i, pid in enumerate(tqdm(problem_ids, total=len(problem_ids))):
        if pid in ignore_problems:
            continue

        folder_name = 'Java' if args.lang == 'java' else 'Python'
        file_ext = 'java' if args.lang == 'java' else 'py'
        pid_dir = os.path.join(data_dir, pid, folder_name)
        meta_file = os.path.join(meta_data_dir, '{}.csv'.format(pid))
        meta_data = read_meta_data(meta_file)

        for file in glob.glob("{}/*.{}".format(pid_dir, file_ext)):
            submission_id = os.path.splitext(os.path.basename(file))[0]
            if submission_id in outliers:
                continue
            if submission_id not in singletons:
                continue

            with open(file) as f:
                content = f.read()
                if content:
                    if args.category != 'all' and meta_data[submission_id] != args.category:
                        continue
                    data.append({
                        "problem_id": pid,
                        "submission_id": submission_id,
                        "code": content
                    })

    with open(
            '{}/{}_train_0.jsonl'.format(args.target_dir, args.lang), 'w', encoding='utf-8'
    ) as writer:
        writer.write('\n'.join([json.dumps(ex) for ex in data]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        help='Language name',
        choices=["java", "python"],
    )
    parser.add_argument(
        "--source_dir", help='Source directory',
    )
    parser.add_argument(
        "--target_dir", help="Output directory to save extracted functions",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "accepted"],
        help='Filter functions based on status'
    )
    args = parser.parse_args()
    prepare(args)
