#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: Leaper
"""
# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: Leaper
"""
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ..SentenceTransformer import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class MLMLoss(nn.Module):

    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int = 30522,
                 loss_fct: Callable = nn.CrossEntropyLoss(ignore_index=-100)):
        super(MLMLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.mlm = nn.Linear(sentence_embedding_dimension, len(self.model._first_module().tokenizer))
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: [Dict,Tensor]):
        reps = [self.model(sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]
        rep_a = reps[0]
        mlm = self.mlm(rep_a)
        mlm_label = labels["mlm_label"].to(self.model._target_device)
        loss_mlm = self.loss_fct(mlm.view(-1, len(self.model._first_module().tokenizer)), mlm_label.view(-1))
        loss = loss_mlm
        return loss

