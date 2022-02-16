import datetime
import json
import math
import os
import random
import time
import pprint
import string

from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import random_split
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import json
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup

from preprocessor import ASTE_Dataset
from utils import correct_spaces, get_f1_for_trainer

def custom_print(*msg):
    
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            custom_logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            custom_logger.write(str(msg[i]))

def initialise_args():

    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='15res', type=str, required=True,
                        help="[15res, 14res, 16res, lap14]")
    parser.add_argument("--train_dataset_path", default='train', type=str, required=True,
                        help="path to train file")
    parser.add_argument("--dev_dataset_path", default='dev', type=str, required=True,
                        help="path to dev file ")
    parser.add_argument("--test_dataset_path", default='test', type=str, required=True,
                        help="path to test file ")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run direct eval on the dev/test set.")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--logger_name", default = 'logs.txt')
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    task_dir = f"./outputs/{args.task}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    output_dir = f"{task_dir}"

    args.output_dir = output_dir

    return args

def get_dataset(tokenizer, type_path, args):
    return ASTE_Dataset(tokenizer=tokenizer, data_dir=args.dataset_path,
     task=args.task, max_len=args.max_seq_length)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.tokenizer.add_tokens(['<triplet>', '<opinion>', '<sentiment>'], special_tokens = True)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

        

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs, 'progress_bar': logs}

    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))

def evaluate(data_loader, model):

    model.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        outs = model.model.generate(input_ids=batch['source_ids'].to('cuda'), 
                                    attention_mask=batch['source_mask'].to('cuda'), 
                                    max_length=128)
        for i in range(len(outs)):
            dec = tokenizer.decode(outs[i], skip_special_tokens=False)
            labels = np.where(batch["target_ids"][i] != -100, batch["target_ids"][i], tokenizer.pad_token_id)
            target = tokenizer.decode(labels, skip_special_tokens=False)

            outputs.extend(dec)
            targets.extend(target)

    
    decoded_labels = correct_spaces(targets)
    decoded_preds = correct_spaces(outputs)

    p, r, f = get_f1_for_trainer(decoded_preds, decoded_labels )
    _, _, opinion_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'opinion')
    _, _, aspect_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'aspect')
    _, _, sentiment_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'sentiment')

    return {'f1':f, 'prec': p, 'rec': r , 'opinion': opinion_f, 'aspect': aspect_f, 'sentiment': sentiment_f }

if __name__ == '__main__':

    args = initialise_args()
    seed_everything(args.seed)

    sent_map = {}
    sent_map['POS'] = 'positive'
    sent_map['NEU'] = 'neutral'
    sent_map['NEG'] = 'negative'

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    logger = logging.getLogger(__name__)
    custom_logger = logger = open(os.path.join(args.output_dir, args.logger_name), 'w')

    if args.do_train:
        custom_print("\n****** Conduct Training ******")
        
        model = T5FineTuner(args)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=5
        )

        train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        #amp_level='O1',
        max_epochs=args.num_train_epochs,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        # model.model.save_pretrained(args.output_dir)

        custom_print("Finish training and saving the model!")

    
    
    if args.do_eval:

        custom_print("\n****** Conduct Evaluating ******")

        # model = T5FineTuner(args)
        dev_results, test_results = {}, {}
        best_f1, best_checkpoint, best_epoch = -999999.0, None, None
        all_checkpoints, all_epochs = [], []

        saved_model_dir = args.output_dir
        for f in os.listdir(saved_model_dir):
            file_name = os.path.join(saved_model_dir, f)
            if 'cktepoch' in file_name:
                all_checkpoints.append(file_name)

        print(f"We will perform validation on the following checkpoints: {all_checkpoints}")

        
            # load dev and test datasets
        dev_dataset = ASTE_Dataset(tokenizer, data_dir=args.dev_dataset_path, task=args.task, max_len=args.max_seq_length)
        dev_loader = DataLoader(dev_dataset, batch_size=32)

        test_dataset = ASTE_Dataset(tokenizer, data_dir=args.test_dataset_path, task=args.task, max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32)

        for checkpoint in all_checkpoints:
            epoch = checkpoint.split('=')[-1][:-5] if len(checkpoint) > 1 else ""
            # only perform evaluation at the specific epochs ("15-19")
            # eval_begin, eval_end = args.eval_begin_end.split('-')
            if 0 <= int(epoch) < 100:
                all_epochs.append(epoch)

                # reload the model and conduct inference
                print(f"\nLoad the trained model from {checkpoint}...")
                model_ckpt = torch.load(checkpoint)
                tuner = T5FineTuner(model_ckpt['hyper_parameters'])
                tuner.load_state_dict(model_ckpt['state_dict'])
                model = tuner.model
                model.to('cuda')
                
                dev_result = evaluate(dev_loader, model)
                if dev_result['f1'] > best_f1:
                    best_f1 = dev_result['f1']
                    best_checkpoint = checkpoint
                    best_epoch = epoch

                # add the global step to the name of these metrics for recording
                # 'f1' --> 'f1_1000'
                dev_result = dict((k + '_{}'.format(epoch), v) for k, v in dev_result.items())
                dev_results.update(dev_result)

                test_result = evaluate(test_loader, model)
                test_result = dict((k + '_{}'.format(epoch), v) for k, v in test_result.items())
                test_results.update(test_result)

    # print test results over last few steps
        custom_print(f"\n\nThe best checkpoint is {best_checkpoint}")
        best_step_metric = f"f1_{best_epoch}"
        custom_print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")

        custom_print("\n* Results *:  Dev  /  Test  \n")
        metric_names = ['f1', 'prec', 'rec']
        for epoch in all_epochs:
            custom_print(f"Epoch-{epoch}:")
            for name in metric_names:
                name_step = f'{name}_{epoch}'
                print(f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}", sep='  ')
            custom_print()





    custom_print("All Done :)")
    custom_logger.close()
    



