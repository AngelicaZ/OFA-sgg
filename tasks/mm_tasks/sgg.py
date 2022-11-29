
from curses import meta
from dataclasses import dataclass, field
import json
import logging
import os
import math
import pickle
from typing import Optional
from argparse import Namespace
from data.file_dataset import FileDataset
import pdb
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F
import sacrebleu
import torch
from fairseq.tasks import register_task
import string
from fairseq import metrics, utils
from fairseq.optim.amp_optimizer import AMPOptimizer
from sacrebleu.metrics import BLEU

from models import search
from data.mm_data.sgg_GQA_dataset import SggGQADataset, GQADatasetReader
from data.mm_data.sgg_VG_dataset import SggVGDataset, VGDatasetReader
from data.mm_data.sg_raw import GQASceneDataset, load_json, np_load
from tasks.ofa_task import OFAConfig, OFATask
from utils.trie import Trie
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD

EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)

@dataclass
class SggConfig(OFAConfig):
    img_dir: str = field(
        default='', metadata={"help": "image for scene graph generation"}
    )
    embedding_json_path: str = field(
        default='', metadata={"help": "embedding json path, for GQA"}
    )
    base_dir: str = field(
        default='', metadata={"help": "base directory for sgg features h5 and json file, for GQA"}
    )
    sgg_features_h5_file: str = field(
        default='', metadata={"help": "sgg features h5 file path, for GQA"}
    )
    sgg_features_json_file: str = field(
        default='', metadata={"help": "sgg features jason file, for GQA"}
    )
    vocab_json_path: str = field(
        default='', metadata={"help": "vocabulary jason path, for GQA"}
    )
    new_vocab_json_path: str = field(
        default='', metadata={"help": "new vocabulary with labels, attr, relationship, obj_x, R_a_b, etc., for GQA"}
    )
    roidb_file: str = field(
        default='', metadata={"help": "scenegraph h5 file, for VG"}
    )
    dict_file: str = field(
        default='', metadata={"help": "dict file, for VG"}
    )
    image_file: str = field(
        default='', metadata={"help": "image info file, for VG"}
    )
    dataset_choose: str = field(
        default='GQA', metadata={"help": "choose dataset to use for scene graph genreation, should be GQA or VG"}
    )
    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    max_obj_num: int = field(
        default=100, metadata={"help": "max number of objects in an image"}
    )
    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_cider: bool = field(
        default=False, metadata={"help": "evaluation with CIDEr scores"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    eval_cider_cached_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"},
    )

    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    tgt_seq_len: int = field(
        default=350, metadata={"help": "the required length of target sequence"}
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )


@register_task("sgg", dataclass=SggConfig)
class SggTask(OFATask):
    def __init__(self, cfg: SggConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        # src_dict: <fairseq.data.dictionary.Dictionary object at 0x7fb93f721b90>
        # tgt_dict: <fairseq.data.dictionary.Dictionary object at 0x7fb93fb15150>

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        print("loading dataset...")
        # print("split: ", split)
        
        if self.cfg.dataset_choose == 'GQA':
            
            paths = self.cfg.data.split(',')
            assert len(paths) > 0
            img_dir = self.cfg.img_dir
            sgg_features_h5_file = self.cfg.sgg_features_h5_file
            sgg_features_json_file = self.cfg.sgg_features_json_file
            embedding_json_path = self.cfg.embedding_json_path
            vocab_json_path = self.cfg.vocab_json_path
            new_vocab_json_path = self.cfg.new_vocab_json_path
            tgt_seq_len = self.cfg.tgt_seq_len
            num_bins = self.cfg.num_bins

            if split == 'train':
                scenegraphs_json_path = paths[0]
            else:
                scenegraphs_json_path = paths[1]
            scenegraphs_json = load_json(scenegraphs_json_path)
            dataset = GQADatasetReader(
                scenegraphs_json,
                img_dir,
                sgg_features_json_file,
                sgg_features_h5_file,
                embedding_json_path,
                vocab_json_path,
                new_vocab_json_path,
                tgt_seq_len,
                num_bins)
            
            self.datasets[split] = SggGQADataset(
                split,
                dataset,
                self.bpe,
                self.src_dict,
                self.tgt_dict
            )
        
        elif self.cfg.dataset_choose == 'VG':

            img_dir = self.cfg.img_dir
            roidb_file = self.cfg.data
            dict_file = self.cfg.dict_file
            image_file = self.cfg.image_file

            tgt_seq_len = self.cfg.tgt_seq_len

            dataset = VGDatasetReader(
                split, 
                img_dir, 
                roidb_file, 
                dict_file, 
                image_file,
                self.bpe,
                num_im=-1,
                num_val_im=5000,
                required_len=tgt_seq_len
            )

            self.datasets[split] = SggVGDataset(
                split,
                dataset,
                self.bpe,
                self.src_dict,
                self.tgt_dict,
                img_dir
            )

        else:
            raise NotImplementedError

    def build_model(self, cfg):
        print("building model...")
        '''
        model in models/ofa/unify_transformer.py -> fairseq/models
        class TransformerModel -> FairseqEncoderDecoderModel -> fairseq_model.py/BaseFairseqModel(nn.Module)
        class TransformerEncoder -> FairseqEncoder
        class TransformerDecoder -> FairseqIncrementalDecoder
        '''
        model = super().build_model(cfg)

        gen_args = json.loads(self.cfg.eval_args)
        self.sequence_generator = self.build_generator(
            [model], Namespace(**gen_args)
        )

        return model
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, **extra_kwargs
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, update_num=update_num) # self.sequence_generator, 
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def _calculate_ap_score(self, hyps, refs, thresh=0.3):
        '''
        hyps shape: [16, 4]
        refs shape: [16, 4]
        '''
        try:
            assert hyps.shape[0] == refs.shape[0]
        except:
            print("hyps shape in _calculate_ap_score: ", hyps.shape)
            print("refs shape in _calculate_ap_score: ", refs.shape)
            print("hyps in _calculate_ap_score: ", hyps)
            print("refs in _calculate_ap_score: ", refs)
        interacts = torch.cat( # the intersect area
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample) # self.sequence_generator, 

        model.eval()
        if self.cfg.eval_bleu:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            bleu = sacrebleu.corpus_bleu(hyps, list(zip_longest(*refs)))
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

        if self.cfg.eval_acc:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            hyps = hyps / (self.cfg.num_bins - 1) * self.cfg.max_image_size
            refs = refs / (self.cfg.num_bins - 1) * self.cfg.max_image_size
            hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
            refs[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            refs[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

            hyps = hyps.reshape(-1, 4)
            refs = refs.reshape(-1, 4)

            scores = self._calculate_ap_score(hyps, refs)
            # scores = self._calculate_ap_score(hyps, sample['region_coords'].float())
            logging_output["_score_sum"] = scores.sum().item()
            logging_output["_score_cnt"] = scores.size(0)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        """Aggregate logging outputs from data parallel training."""

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result
        
        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if self.cfg.eval_bleu:
            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(BLEU.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = BLEU.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2) # TODO: modify to trunc or floor

                metrics.log_derived("bleu", compute_bleu) # Log a scalar value derived from other meters.
        
        if self.cfg.eval_acc:
            if sum_logs("_score_cnt") > 0:
                metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
                metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
                metrics.log_derived("score", compute_score)

    def _inference(self, generator, sample, model):

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                # unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        # print("gen_out: ", gen_out)
        '''
        inference_step: FairseqTask -> generator.generate(models, sample, prefix_tokens=prefix_tokens, constraints=constraints)
        generator: OFATask(FairseqTask) -> build_generator -> models/sequence_generator -> SequenceGenerator -> _generate(models, tgt_dict)
        tgt_dict: 
        '''
        hyps, refs = [], []
        # print("length of reference: ", len(gen_out))
        # transtab = str.maketrans({key: None for key in string.punctuation}) # TODO: this line deal with the punctuation

        if self.cfg.eval_acc:
            max_len = 0
            for i in range(len(gen_out)):
                ref = sample["target"][i][:-1] - len(self.src_dict) + self.cfg.num_bins
                hyp = gen_out[i][0]["tokens"][:-1] - len(self.src_dict) + self.cfg.num_bins
                ref_list = ref.tolist()
                hyp_list = hyp.tolist()
                # print("ref: ", ref)
                # print("hyp: ", hyp)
                hyp_i = []
                ref_i = []
                for j in range(len(ref_list)):
                    if ref[j] == -58454:
                        try: 
                            hyp_i.extend([hyp_list[j-4], hyp_list[j-3], hyp_list[j-2], hyp_list[j-1]])
                            ref_i.extend([ref_list[j-4], ref_list[j-3], ref_list[j-2], ref_list[j-1]])
                        except:
                            print("hyp_i: ", hyp_list)
                            print("ref_i: ", ref_list)
                if len(ref_i) > max_len:
                    max_len = len(ref_i)
                hyps.append(hyp_i) # torch.tensor(hyp_i).to(ref.device)
                refs.append(ref_i)
            for i in range(len(refs)):
                if len(hyps[i]) < max_len:
                    for j in range(max_len - len(hyps[i])):
                        hyps[i].append(0)
                if len(refs[i]) < max_len:
                    for j in range(max_len - len(refs[i])):
                        refs[i].append(0)
                hyps[i] = torch.tensor(hyps[i]).to(ref.device)
                refs[i] = torch.tensor(refs[i]).to(ref.device)
                
        if self.cfg.eval_bleu:
            for i in range(len(gen_out)):
                decode_tokens = decode(gen_out[i][0]["tokens"])
                decode_tokens_clean = decode_tokens.replace('&&', ' ')
                print("decode_tokens: ", decode_tokens_clean)
                # hyps.append(decode_tokens.translate(transtab).strip())
                hyps.append(decode_tokens)
                refs.append(
                    [
                        # sent.translate(transtab).strip()
                        sent.strip()
                        for sent in decode(
                            utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                            escape_unk=True,  # don't count <unk> as matches to the hypo
                        ).split(' ')
                    ]
                )
            
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' '.join(refs[0]))

        # return hyps, refs
        try:
            assert len(hyps) == len(refs)
        except:
            print("hyps shape in _inference: ", hyps.shape)
            print("refs shape in _inference: ", refs.shape)
            print("hyps in _inference: ", hyps)
            print("refs in _inference: ", refs)
        return torch.stack(hyps, dim=0), torch.stack(refs, dim=0)