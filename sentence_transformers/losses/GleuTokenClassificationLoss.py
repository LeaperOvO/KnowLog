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


class GleuTokenClassificationLoss(nn.Module):

    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 loss_fct: Callable = nn.CrossEntropyLoss(ignore_index=-100)):
        super(GleuTokenClassificationLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.classifier = nn.Linear(sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['token_embeddings'] for sentence_feature in sentence_features]
        rep_a = reps[0]
        # rep_a = rep_a[:,1,:]
        output = self.classifier(rep_a)
        if labels is not None:
            loss = self.loss_fct(output.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return reps, output
