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
    args.add_argument("-train_data", "--train_data", type=str,
                      default="./datasets/tasks/MC/hw_switch_train.json", help="train dataset")

    args.add_argument("-dev_data", "--dev_data", type=str,
                      default="./datasets/tasks/MC/hw_switch_dev.json", help="dev dataset")

    args.add_argument("-test_data", "--test_data", type=str,
                      default="./datasets/tasks/MC/hw_switch_test.json", help="test dataset")

    args.add_argument("-pretrain_model", "--pretrain_model", type=str,
                      default="bert-base-uncased", help="the path of the pretrained model to finetune")


    args.add_argument("-epoch", "--epoch", type=int,
                      default=30, help="Number of epochs")

    args.add_argument("-batch_size", "--batch_size", type=int,
                      default=8, help="Batch Size")

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default="./output/knowlog_finetune", help="Folder name to save the models.")

    args = args.parse_args()
    return args

def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content

def evaluate(args):
    model_save_path = args.outfolder
    train_batch_size = args.batch_size
    num_epochs = args.epoch

    train_data = read_json(args.train_data)
    dev_data = read_json(args.dev_data)
    test_data = read_json(args.test_data)

    # load model
    model = SentenceTransformer(args.pretrain_model)

    # load dataset
    train_samples = []
    dev_samples = []
    test_samples = []

    label_count = -1

    for item in train_data:
        if item[1] > label_count:
            label_count = item[1]
        train_samples.append(InputExample(texts=[item[0]], label=item[1]))
    for item in test_data:
        if item[1] > label_count:
            label_count = item[1]
        test_samples.append(InputExample(texts=[item[0]], label=item[1]))
    for item in dev_data:
        if item[1] > label_count:
            label_count = item[1]
        dev_samples.append(InputExample(texts=[item[0]], label=item[1]))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)

    # loss
    train_loss = losses.SingleSoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=label_count)

    dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name='test2')
    test_evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss, name='test2')

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # finetune and evaluate
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              evaluator2=test_evaluator,
              epochs=num_epochs,
              evaluation_steps=10000,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              )

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
