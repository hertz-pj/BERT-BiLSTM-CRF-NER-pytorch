from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys
import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from utils import NerProcessor, convert_examples_to_features, get_Dataset
from models import BERT_BiLSTM_CRF

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

logger = logging.getLogger(__name__)

# set the random seed for repeat
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def evaluate(args, ):


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--eval_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    ## other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_train", default=False, type=boolean_string)
    parser.add_argument("--do_eval", default=False, type=boolean_string)
    parser.add_argument("--do_test", default=False, type=boolean_string)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--use_weight", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--logging_steps", default=500, type=int)

    parser.add_argument("--need_birnn", default=False, type=boolean_string)
    parser.add_argument("--rnn_dim", default=128, type=int)

    parser.add_argument("--gpu_id", default="0", type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    logger.info(f"device: {device} n_gpu: {n_gpu}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    # now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H')
    # tmp_dir = args.output_dir + '/' +str(now_time) + '_ernie'
    # if not os.path.exists(tmp_dir):
    #     os.makedirs(tmp_dir)
    # args.output_dir = tmp_dir

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, "eval")):
        os.makedirs(os.path.join(args.output_dir, "eval"))
    
    writer = SummaryWriter(logdir=os.path.join(args.output_dir, "eval"), comment="Linear")


    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Prepare optimizer and schedule (linear warmup and decay)

    if args.do_train:

        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, 
                    do_lower_case=args.do_lower_case)
        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, 
                num_labels=num_labels)

        model = BERT_BiLSTM_CRF.from_pretrained(args.model_name_or_path, config=config, 
                need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)

        model.to(device)
        
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        train_examples, train_features, train_data = get_Dataset(args, processor, tokenizer, mode="train")
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.eval:
            eval_examples, eval_features, eval_data = get_Dataset(args, processor, tokenizer, mode="eval")

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Total optimization steps = %d", t_total)

        model.train()
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = model(input_ids, label_ids, segment_ids, input_mask)
                loss = outputs

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        tr_loss_avg = (tr_loss-logging_loss)/args.logging_steps
                        writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                        logging_loss = tr_loss

            logger.info(f'epoch {ep}, train loss: {tr_loss}')
        # writer.add_graph(model)
        writer.close()

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


    if args.do_test:
        # model = BertForTokenClassification.from_pretrained(args.output_dir)
        # model.to(device)
        label_map = {i : label for i, label in enumerate(label_list)}

        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = BERT_BiLSTM_CRF.from_pretrained(args.output_dir)
        model.to(device)

        eval_examples = processor.get_examples(args.test_file)
        eval_features = convert_examples_to_features(
            args, eval_examples, label_list, args.max_seq_length, tokenizer
        )

        logger.info("***** Running test *****")
        logger.info(f" Num examples = {len(eval_examples)}")
        logger.info(f" Batch size = {args.eval_batch_size}")

        all_ori_tokens = [f.ori_tokens for f in eval_features]
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()

        pred_labels = []
        
        for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            tr_loss_total = 0
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model.predict(input_ids, segment_ids, input_mask)
            # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            # logits = logits.detach().cpu().numpy()

            for l in logits:

                pred_label = []
                for idx in l:
                    pred_label.append(label_map[idx])
                pred_labels.append(pred_label)

        assert len(pred_labels) == len(all_ori_tokens)
        print(len(pred_labels))
        with open(os.path.join(args.output_dir, "token_labels_.txt"), "w", encoding="utf-8") as f:
            for ori_tokens, prel in zip(all_ori_tokens, pred_labels):
                for ot,pl in zip(ori_tokens, prel):
                    if ot in ["[CLS]", "[SEP]"]:
                        continue
                    else:
                        f.write(f"{ot} O {pl}\n")
                f.write("\n")

if __name__ == "__main__":
    main()
    pass