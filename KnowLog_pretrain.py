import argparse
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,LabelAccuracyEvaluator,BinaryClassificationEvaluator
from torch.utils.data import DataLoader
import logging
import json
import random
import os
import sys
import math
import numpy as np

random.seed(1)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-pretrain_data", "--pretrain_data", type=str,
                      default="./datasets/pre-train/all_log.json", help="pre-train data directory")

    args.add_argument("-abbr", "--abbr", type=str,
                      default="./datasets/pre-train/abbr.json", help="abbreviations directory")

    args.add_argument("-vocab", "--vocab", type=bool,
                      default="True", help="Whether abbreviations join the vocabulary")

    args.add_argument("-base_model", "--base_model", type=str,
                      default="bert-base-uncased", help="base_model")

    args.add_argument("-p", "--p", type=float,
                      default=0.5, help="probability of masking abbr")

    args.add_argument("-epoch", "--epoch", type=int,
                      default=50, help="Number of epochs")

    args.add_argument("-batch_size", "--batch_size", type=int,
                      default=8, help="Batch Size")

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default="./output/knowlog-bert", help="Folder name to save the models.")

    args = args.parse_args()
    return args

def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content

def IsEnglish(character):
    for cha in character:
        if not 'A' <= cha <= 'Z':
            return False
    else:
        return True

def train(args):
    datapath = args.pretrain_data
    model_save_path = args.outfolder
    train_batch_size = args.batch_size
    num_epochs = args.epoch

    # load data
    data = read_json(datapath)

    # load model
    model_name = args.base_model
    word_embedding_model = models.Transformer(model_name)

    # add abbr to vocab
    abbr_list = read_json(args.abbr)
    if args.vocab:
        word_embedding_model.tokenizer.add_tokens(abbr_list, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # encode description
    desc = np.array(data)[:,1].tolist()
    model_nlp = SentenceTransformer(modules=[models.Transformer(model_name), pooling_model])
    desc_embedding = model_nlp.encode(desc, device='cuda', convert_to_numpy=False, convert_to_tensor=True,
                                      batch_size=128)
    con_samples = []
    for i in range(len(data)):
        con_samples.append(InputExample(texts=[data[i][0], data[i][0]], embedding=desc_embedding[i]))

    random.shuffle(con_samples)

    # mask abbr
    p = args.p
    token_samples = []
    for i in range(len(data)):
        if (data[i][0].split('/')[0] in abbr_list and IsEnglish(data[i][0].split('/')[0])):

            if random.random() < p:
                token_samples.append(InputExample(texts=[data[i][0]], label=abbr_list.index(data[i][0].split('/')[0])))
            else:
                s = '[MASK]/' + '/'.join(data[i][0].split('/')[1:])
                token_samples.append(InputExample(texts=[s], label=abbr_list.index(data[i][0].split('/')[0])))
        elif (data[i][0].split('-')[0] in abbr_list and IsEnglish(data[i][0].split('-')[0])):
            if random.random() < p:
                token_samples.append(InputExample(texts=[data[i][0]], label=abbr_list.index(data[i][0].split('-')[0])))
            else:
                s = '[MASK]-' + '-'.join(data[i][0].split('-')[1:])
                token_samples.append(InputExample(texts=[s], label=abbr_list.index(data[i][0].split('-')[0])))

    #   build dataset
    train_con_dataloader = DataLoader(con_samples, shuffle=True, batch_size=train_batch_size)
    train_token_dataloader = DataLoader(token_samples, shuffle=True, batch_size=train_batch_size)

    # loss
    train_con_loss = losses.LogNLMultipleNegativesRankingLoss(model)
    train_token_loss = losses.TokenClassificationLoss(model=model,
                                                      sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                                      num_labels=len(abbr_list))


    warmup_steps = math.ceil(len(train_con_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # train
    model.fit(train_objectives=[(train_con_dataloader, train_con_loss),(train_token_dataloader,train_token_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              save_f=True
              )

if __name__ == '__main__':
    args = parse_args()
    train(args)