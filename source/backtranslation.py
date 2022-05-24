# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from fairseq import metrics, options, models, utils, checkpoint_utils
from fairseq.data import (
    FairseqDataset,
    NoisingDataset,
    PrependTokenDataset,
    LanguagePairDataset,
    TransformEosLangPairDataset,
    AppendTokenDataset,
    RoundRobinZipDatasets,
    data_utils,
    encoders,
)
from fairseq.sequence_generator import SequenceGenerator
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from .utils import (
    load_langpair_dataset,
    submit_functions,
)
from pathlib import Path
from lang_processors.java_processor import JavaProcessor
from lang_processors.cpp_processor import CppProcessor
from lang_processors.python_processor import PythonProcessor
from lang_processors.lang_processor import LangProcessor
from evaluation.utils import TREE_SITTER_ROOT
from concurrent.futures import ProcessPoolExecutor

EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)


@register_task("backtranslation")
class BackTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', metavar='LANGS',
                            help='languages to be included in dictionary')
        parser.add_argument('--mono-langs', default=None, metavar='MONO_LANGS',
                            help='monolingual languages for training')
        parser.add_argument('--inter-lang', default=None,
                            help='intermediate language to use for backtranslation')
        parser.add_argument('--valid-lang-pairs', default=None, metavar='VALID_LANG_PAIRS',
                            help='language pairs for validation')
        parser.add_argument('--inter-trans-steps', type=int, default=0,
                            help='number of steps to perform intermediate translation')
        parser.add_argument('--show-samples-interval', type=int, default=1000,
                            help='interval for showing backtranslation samples')
        parser.add_argument('--show-eval-samples-interval', type=int, default=100,
                            help='interval for showing bleu eval samples')
        parser.add_argument('--target-to-source-checkpoint', type=str, default=None,
                            help='a checkpoint to load separately for the target-to-source translation')
        # options for reporting COMP ACC during validation
        parser.add_argument('--eval-comp-acc', action='store_true',
                            help='evaluation with Computational Accuracy')
        parser.add_argument('--scripts-folder', type=str,
                            help='directory path where G4G scripts located')
        parser.add_argument('--output-scripts-folder', type=str,
                            help='output directory where generated programs will be stored')
        parser.add_argument('--id-folder', type=str,
                            help='directory path where id files are located')

    def __init__(self, args, common_dict, mono_langs, valid_lang_pairs, training):
        super().__init__(args, common_dict, common_dict)
        self.common_dict = common_dict
        self.mono_langs = mono_langs
        self.inter_lang = args.inter_lang
        self.valid_lang_pairs = valid_lang_pairs
        self.training = training

        self.SHOW_SAMPLES_INTERVAL = args.show_samples_interval
        self._show_samples_ctr = {l: self.SHOW_SAMPLES_INTERVAL for l in self.mono_langs}
        self.SHOW_EVAL_SAMPLES_INTERVAL = args.show_eval_samples_interval
        self._show_eval_samples_ctr = {p: self.SHOW_EVAL_SAMPLES_INTERVAL for p in self.valid_lang_pairs}
        self.SHOW_SAMPLES_NUMBER = 1

        # for COMP ACC validation
        self.lang_processor = {}
        self.valid_function_map = {}

        if self.args.eval_comp_acc:
            assert self.args.eval_bleu
            output_scripts_folder = {}
            for lang_pair in valid_lang_pairs:
                src, tgt = lang_pair.split("-")
                self.lang_processor[tgt] = LangProcessor.processors[tgt](root_folder=TREE_SITTER_ROOT)
                output_scripts_folder[tgt] = os.path.join(self.args.output_scripts_folder, tgt)
                Path(output_scripts_folder[tgt]).mkdir(parents=True, exist_ok=True)
            self.args.output_scripts_folder = output_scripts_folder

        # self.args = args
        self.data = self.args.data.split(":")
        if len(self.data) == 1:
            shards = list(Path(self.data[0]).glob("shard*"))
            if len(shards) > 0:
                # keep this as strings, since it can also be a manifold path
                old_data = self.data
                self.data = [str(shard) for shard in shards]
                logging.warning(f"Expanded data directory {old_data} to {self.data}")

        # load a translation model
        self.target_to_source_model = None
        if self.training and args.target_to_source_checkpoint:
            self.target_to_source_model = super().build_model(args)
            state = checkpoint_utils.load_checkpoint_to_cpu(args.target_to_source_checkpoint)
            self.target_to_source_model.load_state_dict(state["model"], strict=True)
            logger.info(
                "loaded checkpoint {} (epoch {})".format(
                    args.target_to_source_checkpoint,
                    state["extra_state"]["train_iterator"]["epoch"]
                )
            )
            is_cuda = torch.cuda.is_available() and not args.cpu
            device = torch.device("cuda") if is_cuda else torch.device("cpu")
            if args.fp16:
                self.target_to_source_model = self.target_to_source_model.half()
            if is_cuda:
                self.target_to_source_model = self.target_to_source_model.to(device=device)
            if (
                    args.distributed_world_size > 1
                    and not args.use_bmuf
            ):
                self.target_to_source_model = models.DistributedFairseqModel(
                    args,
                    self.target_to_source_model,
                    process_group=None,
                )
            self.target_to_source_model.eval()

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = args.data.split(":")
        assert len(paths) > 0

        langs = args.langs.split(",")
        mono_langs = args.mono_langs
        valid_lang_pairs = args.valid_lang_pairs

        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        if training:
            mono_langs = mono_langs.split(",")
            valid_lang_pairs = valid_lang_pairs.split(",")
            assert args.inter_lang in langs
            assert args.inter_lang not in mono_langs
        else:
            mono_langs = []
            valid_lang_pairs = []

        # load dictionary
        dict_path = os.path.join(paths[0], "dict.txt")
        common_dict = cls.load_dictionary(dict_path)

        # to match PLBART
        for l in langs:
            common_dict.add_symbol(_lang_token(l))
        # NOTE: when we fine-tune PLBART in multilingual code summarization
        #  or generation, the <mask> token is dropped

        return cls(args, common_dict, mono_langs, valid_lang_pairs, training)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs) -> FairseqDataset:
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split == "train":
            data_path = self.data[(epoch - 1) % len(self.data)]
            dataset = self.load_train_dataset(data_path)
        elif split == "valid" and self.training:
            valid_datasets = []

            def split_exists(split, src, tgt, data_path):
                filename = os.path.join(data_path, "{}.{}-{}.id".format(split, src, tgt))
                return os.path.exists(filename)

            for lang_pair in self.valid_lang_pairs:
                src, tgt = lang_pair.split("-")
                dataset = self.load_translation_dataset(split, self.data[0], src, tgt)
                valid_datasets.append((f"{lang_pair}", dataset))

                if self.args.eval_comp_acc:
                    if split_exists(split, src, tgt, self.args.id_folder):
                        prefix = os.path.join(self.args.id_folder, "{}.{}-{}.".format(split, src, tgt))
                    else:
                        assert split_exists(split, tgt, src, self.args.id_folder)
                        prefix = os.path.join(self.args.id_folder, "{}.{}-{}.".format(split, tgt, src))

                    ex_ids = open(prefix + 'id').read().splitlines()
                    self.valid_function_map[tgt] = {j: eid for j, eid in enumerate(ex_ids)}

            dataset = RoundRobinZipDatasets(OrderedDict(valid_datasets))
        else:
            src, tgt = self.args.source_lang, self.args.target_lang
            dataset = self.load_translation_dataset(split, self.data[0], src, tgt)

        self.datasets[split] = dataset
        return dataset

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset."""
        data = []
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))

        return RoundRobinZipDatasets(OrderedDict(data))

    def _langpair_dataset(
            self, src: FairseqDataset, tgt: FairseqDataset
    ) -> LanguagePairDataset:
        return LanguagePairDataset(
            src,
            src.sizes,
            self.dictionary,
            tgt=tgt,
            tgt_sizes=tgt.sizes,
            tgt_dict=self.dictionary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
        )

    def _prepend_lang_bos_to_target(
            self, dataset: LanguagePairDataset, lang: str
    ) -> LanguagePairDataset:
        bos = _lang_token_index(self.dictionary, lang)
        return TransformEosLangPairDataset(
            dataset,
            src_eos=self.dictionary.eos(),
            tgt_bos=self.dictionary.eos(),
            new_tgt_bos=bos,
        )

    def load_bt_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """The BT dataset is generated with (tgt, tgt) pairs.
        The actual translation to a (generated_src, tgt) pair
        is done on the fly during training.
        """
        mono_dataset = data_utils.load_indexed_dataset(
            data_path, self.common_dict, self.args.dataset_impl
        )
        assert mono_dataset is not None, f"No dataset found for {lang}"

        # Note that, PLBART appends the language id
        mono_dataset_src = AppendTokenDataset(
            mono_dataset, _lang_token_index(self.dictionary, lang)
        )

        mono_dataset_bt = self._langpair_dataset(mono_dataset_src, mono_dataset)
        logger.info(
            f"mono_lang = {lang} "
            f"lang token index = {_lang_token_index(self.dictionary, lang)} "
            f"lang token = {_lang_token(lang)}"
        )

        mono_dataset_bt = self._prepend_lang_bos_to_target(mono_dataset_bt, lang)
        return mono_dataset_bt

    def load_translation_dataset(
            self, split: str, data_path: str, src: str, tgt: str, combine: bool = False
    ):
        """
        * Each instance in translation dataset looks like below.
        * At the start of this function, `smp` has the same input and target:
          |-----------------------------------------------------------|
          | print hello world [en_XX] | [python] print("hello world") |
          |-----------------------------------------------------------|
        """

        # use the same function than TranslationTask
        src_tgt_dt = load_langpair_dataset(
            data_path,
            split,
            src,
            self.common_dict,
            tgt,
            self.common_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            append_bos_src=_lang_token_index(self.dictionary, src),
        )

        src_tgt_eos_dt = self._prepend_lang_bos_to_target(src_tgt_dt, tgt)
        src_tgt_eos_dt.args = self.args
        return src_tgt_eos_dt

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        src_lang_id = self.source_dictionary.index(_lang_token(self.args.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(
            source_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
        )
        return dataset

    def build_model(self, args):
        # torch.autograd.set_detect_anomaly(True)
        model = super().build_model(args)
        if self.training:
            self.translation_generator = SequenceGenerator(
                [model],
                tgt_dict=self.dictionary,
                beam_size=1,
                max_len_b=256,
            )
            self.translate_from_inter_lang = SequenceGenerator(
                [model if self.target_to_source_model is None else self.target_to_source_model],
                tgt_dict=self.dictionary,
                beam_size=1,
                max_len_b=256,
            )
            self.translate_to_inter_lang = SequenceGenerator(
                [model if self.target_to_source_model is None else self.target_to_source_model],
                tgt_dict=self.dictionary,
                beam_size=1,
                max_len_b=64,
                no_repeat_ngram_size=3,
            )

        return model

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.common_dict

    def display_samples_once_in_a_while(self, smp, mono_lang, other_lang, step, inter_generated):
        self._show_samples_ctr[mono_lang] += 1
        if self._show_samples_ctr[mono_lang] < self.SHOW_SAMPLES_INTERVAL:
            return
        self._show_samples_ctr[mono_lang] = 0

        ln = smp["net_input"]["src_tokens"].shape[0]

        if inter_generated is not None:
            logger.info(
                f"(step:{step}) : "
                f"{mono_lang} ---> {self.inter_lang} ---> {other_lang} "
                f"({other_lang} was generated by summarization and generation.) {ln} samples"
            )
        else:
            logger.info(
                f"(step:{step}) : "
                f"{mono_lang} ---> {other_lang} "
                f"({other_lang} was generated by back-translation.) {ln} samples"
            )

        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            src_tokens = src_tokens[src_tokens != self.dictionary.pad()][:-1]
            tgt_tokens = smp["target"][i]

            src_str = self.dictionary.string(src_tokens, "sentencepiece")
            tgt_str = self.dictionary.string(tgt_tokens, "sentencepiece")

            if inter_generated is not None:
                inter_tokens = inter_generated[i]
                inter_tokens = inter_tokens[inter_tokens != self.dictionary.pad()][:-1]
                inter_str = self.dictionary.string(inter_tokens, "sentencepiece")
                logger.info(
                    f"\n{i}\n"
                    f"[{other_lang}-generated]  {src_str}\n"
                    f"[{self.inter_lang}-intermediate]  {inter_str}\n"
                    f"[{mono_lang}-original]  {tgt_str}\n"
                )
            else:
                logger.info(
                    f"\n{i}\n"
                    f"[{other_lang} generated]  {src_str}\n"
                    f"[{mono_lang} original]  {tgt_str}\n"
                )

    def backtranslate_sample(self, sample, orig_lang, other_lang) -> None:
        """
        * WARNING: smp is modified in place.
        * At the start of this function, `smp` has the same input and target:
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (from data) hello world [en]   |  [en] hello world     |
          |--------------------------------------------------------|
        * We call generator.generate(smp, bos_token = token("ro")),
        and copy the result as input
        * At the end, `smp` has the translation to other language.
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (generated) salut lume [ro]    |  [en] hello world     |
          |--------------------------------------------------------|
        """
        bos_token = _lang_token_index(self.dictionary, other_lang)
        with torch.no_grad():
            if orig_lang == self.inter_lang:
                generated = self.translate_from_inter_lang.generate(
                    models=[],
                    sample=sample,
                    bos_token=bos_token,
                )
            elif other_lang == self.inter_lang:
                generated = self.translate_to_inter_lang.generate(
                    models=[],
                    sample=sample,
                    bos_token=bos_token,
                )
            else:
                generated = self.translation_generator.generate(
                    models=[],
                    sample=sample,
                    bos_token=bos_token,
                )

        max_lngth = max([gn[0]["tokens"].size(0) for gn in generated])
        net_input = sample["net_input"]
        n_src_tokens = torch.empty(
            size=(len(generated), max_lngth + 1),
            dtype=net_input["src_tokens"].dtype
        )
        n_src_lengths = torch.empty(
            len(generated),
            dtype=net_input["src_lengths"].dtype
        )

        for i, gn in enumerate(generated):
            tokens = gn[0]["tokens"]
            tokens_size = tokens.size(0)
            padding_needed = max_lngth - tokens_size
            tokens = torch.cat([tokens, tokens.new([bos_token])])
            tokens = F.pad(tokens, (0, padding_needed), value=self.dictionary.pad())
            n_src_tokens[i] = tokens
            n_src_lengths[i] = tokens_size + 1

        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = n_src_tokens.to(device)
        net_input["src_lengths"] = n_src_lengths.to(device)

    def get_other_lang(self, lang):
        select_from_langs = [l for l in self.mono_langs if l != lang]
        num_candidates = len(select_from_langs)
        assert num_candidates in [1, 2]

        if num_candidates == 1:
            return select_from_langs[0]
        else:
            # TODO: should we sample based on uniform probability?
            #  For example, {Java->C++, C++->Java} is comparatively easier than {Python->C++, Python->Java}
            return np.random.choice(select_from_langs)
            # if lang == 'python':
            #     # with equal probability, we select Java or C++ as the source language
            #     return np.random.choice(select_from_langs)
            # else:
            #     # else, Python is selected as source with 0.75 probability
            #     probs = [0.75, 0.25] if select_from_langs[0] == 'python' else [0.25, 0.75]
            #     return np.random.choice(select_from_langs, p=probs)

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
        dataset_keys = self.datasets["train"].datasets.keys()

        for dataset_key in dataset_keys:
            if (
                    dataset_key not in sample
                    or sample[dataset_key] is None
                    or len(sample[dataset_key]) == 0
            ):
                continue
            mono_lang, task_subtype = dataset_key.split("-")
            assert task_subtype == "BT"
            smp = sample[dataset_key]
            with torch.autograd.profiler.record_function("backtranslation"):
                model.eval()
                inter_generated = None
                other_lang = self.get_other_lang(mono_lang)
                if self.inter_lang and update_num < self.args.inter_trans_steps:
                    # pl -> nl, e.g., src_lang = java, tgt_lang = en_XX
                    self.backtranslate_sample(smp, mono_lang, self.inter_lang)
                    inter_generated = smp["net_input"]["src_tokens"].clone().detach()
                    # nl -> pl, e.g., src_lang = en_XX, tgt_lang = python
                    self.backtranslate_sample(smp, self.inter_lang, other_lang)
                else:
                    self.backtranslate_sample(smp, mono_lang, other_lang)

                self.display_samples_once_in_a_while(
                    smp, mono_lang, other_lang, update_num, inter_generated
                )
                if inter_generated is not None:
                    del inter_generated
                model.train()

            # Like in FairseqTask.train_step
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, smp)

            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)

            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        dataset_keys = self.datasets["valid"].datasets.keys()
        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
        model.eval()
        with torch.no_grad():
            for dataset_key in dataset_keys:
                if (
                        dataset_key not in sample
                        or sample[dataset_key] is None
                        or len(sample[dataset_key]) == 0
                ):
                    continue
                smp = sample[dataset_key]
                loss, sample_size, logging_output = criterion(model, smp)
                agg_loss += loss.data.item()
                agg_sample_size += sample_size

                hyps, refs = [], []
                if self.args.eval_bleu:
                    bleu, (hyps, refs) = self.inference_with_bleu(smp, model, dataset_key)
                    logging_output["_bleu_sys_len"] = bleu.sys_len
                    logging_output["_bleu_ref_len"] = bleu.ref_len
                    # we split counts into separate entries so that they can be
                    # summed efficiently across workers using fast-stat-sync
                    assert len(bleu.counts) == EVAL_BLEU_ORDER
                    for i in range(EVAL_BLEU_ORDER):
                        logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                        logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                    agg_logging_output[f"{dataset_key}:{k}"] += logging_output[k]

                if self.args.eval_comp_acc:
                    tgt_lang = dataset_key.split('-')[1]
                    logging_output = defaultdict(float)
                    for i in range(len(hyps)):
                        results_list, _ = submit_functions(
                            self.lang_processor[tgt_lang],
                            functions_list=[hyps[i]],
                            id=self.valid_function_map[tgt_lang][smp["id"][i].item()],
                            ref=refs[i],
                            lang=tgt_lang,
                            script_folder=self.args.scripts_folder,
                            outfolder=self.args.output_scripts_folder[tgt_lang]
                        )
                        nb_success = sum([r[0] == "success" for r in results_list])
                        nb_identical = sum(
                            [r[0] == "success" and r[1] == "identical to gold" for r in results_list]
                        )
                        assert nb_success <= 1, "Should stop after first success"
                        if nb_success > 0:
                            logging_output["success"] += 1
                            if nb_identical > 0:
                                logging_output["identical_gold"] += 1
                        else:
                            logging_output[results_list[0][0]] += 1

                    for k in logging_output:
                        agg_logging_output[f"{dataset_key}:{k}"] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(
            self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                bos_token=self.tgt_dict.index(_lang_token(self.args.target_lang)),
            )

    def inference_with_bleu(self, sample, model, lang_pair):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.generate(self.sequence_generator, model, sample)
        sources, hyps, refs = [], [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
            src_tokens = utils.strip_pad(sample["net_input"]["src_tokens"][i], self.src_dict.pad())
            sources.append(
                decode(src_tokens[:-1], escape_unk=False)
            )

        if self.args.eval_tokenized_bleu:
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            bleu = sacrebleu.corpus_bleu(hyps, [refs])

        if self.args.eval_bleu_print_samples:
            self._show_eval_samples_ctr[lang_pair] += 1
            if self._show_eval_samples_ctr[lang_pair] >= self.SHOW_EVAL_SAMPLES_INTERVAL:
                self._show_eval_samples_ctr[lang_pair] = 0
                logger.info(
                    f"\n{lang_pair}"
                    f"\n[source] {sources[0]}"
                    f"\n[hypothesis] {hyps[0]}"
                    f"\n[reference] {refs[0]}"
                    f"\n[bleu] {bleu.score}"
                )

        return bleu, (hyps, refs)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.args.eval_comp_acc:
            def sum_logs(key):
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            func_run_stats = {
                "success": [],
                "failure": [],
                "error": [],
                "timeout": [],
                "identical_gold": []
            }
            for lang_pair in self.valid_lang_pairs:
                for k in func_run_stats:
                    func_run_stats[k].append(sum_logs(f"{lang_pair}:{k}"))

            successes = func_run_stats["success"]
            totals = [sum(x) for x in zip(*func_run_stats.values())]

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_comp_acc_successes", np.array(successes))
                metrics.log_scalar("_comp_acc_totals", np.array(totals))

                def comp_acc(meters):
                    score = meters["_comp_acc_successes"].sum / meters['_comp_acc_totals'].sum
                    return round(score.mean(), 2)

                metrics.log_derived("comp_acc", comp_acc)

    def generate(self, generator, model, sample):
        bos_token = sample["net_input"]["prev_output_tokens"][0][0]
        with torch.no_grad():
            generated = generator.generate(
                models=[model],
                sample=sample,
                bos_token=bos_token,
            )
        return generated


def _lang_token(lang: str) -> str:
    return f"__{lang}__"


def _guess_lang_token(dictionary, langs: List[str], tokens: torch.Tensor) -> str:
    lang_tokens = [_lang_token(l) for l in langs]
    for t in tokens:
        if dictionary[t] in lang_tokens:
            return dictionary[t]
    return ''


def _lang_token_index(dictionary, lang: str) -> int:
    return dictionary.index(_lang_token(lang))
