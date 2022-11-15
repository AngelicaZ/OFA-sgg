# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import logging
import pdb
import re
import torch.utils.data
from fairseq.data import FairseqDataset, MaskTokensDataset
from fairseq.tokenizer import tokenize_line

logger = logging.getLogger(__name__)


class OFADataset(FairseqDataset):
    def __init__(self, split, dataset, bpe, src_dict, tgt_dict):
        self.split = split
        self.dataset = dataset
        self.bpe = bpe
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.bos = src_dict.bos()
        self.eos = src_dict.eos()
        self.pad = src_dict.pad()
        # self.mask_idx = src_dict.mask_idx()
        # print("self.mask_idx: ", self.mask_idx)
        # pdb.set_trace()
        self.bos_item = torch.LongTensor([self.bos])
        self.eos_item = torch.LongTensor([self.eos])

    def __len__(self):
        return len(self.dataset)

    def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True):
        
        # def encode_line(
        #     line,
        #     line_tokenizer=tokenize_line,
        #     add_if_not_exist=True,
        #     consumer=None,
        #     append_eos=True,
        #     reverse_order=False,
        # ) -> torch.IntTensor:
        #     words = line_tokenizer(line)
        #     print("words: ", words)
        #     if reverse_order:
        #         words = list(reversed(words))
        #     nwords = len(words)
        #     print("nwords: ", nwords)
        #     pdb.set_trace()
        #     ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        #     for i, word in enumerate(words):
        #         if add_if_not_exist:
        #             idx = self.add_symbol(word)
        #         else:
        #             idx = self.index(word)
        #         if consumer is not None:
        #             consumer(word, idx)
        #         ids[i] = idx
        #     if append_eos:
        #         ids[nwords] = self.eos_index
        #     return ids
        
        s = self.tgt_dict.encode_line(
        # s = encode_line(
            line=self.bpe.encode(text) if use_bpe else text,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    def pre_question(self, question, max_ques_words=None):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question

    def pre_caption(self, caption, max_words=None):
        caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if max_words is not None and len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption
