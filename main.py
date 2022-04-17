import datetime
import json
import math
import os
import random
import time
import pprint
import string

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
from transformers.optimization import Adafactor, AdamW
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import json
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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
    parser.add_argument("--model_weights", default='', type=str,
                        help ="In case of external checkpoint weights")                  
    parser.add_argument("--do_train",action='store_true',  help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    # parser.add_argument("--do_direct_eval", action='store_true', 
    #                     help="Whether to run direct eval on the dev/test set.") ## useful when trying zero shot
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--logger_name", default = 'logs.txt')
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--use_tagger", default = False, type = bool) 
    parser.add_argument("--regressor", default = False, type = bool)               
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--k_shot', type=int, default=-1, help="low-resource k shot")

    # training details
    parser.add_argument('--log_message', type=str, default='', help="message to be logged at start")
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

def get_dataset(tokenizer, data_path, task, max_seq_length, k_shot = -1):
    return ASTE_Dataset(tokenizer=tokenizer, data_path = data_path,
     task=task, k_shot = k_shot,  max_len=max_seq_length)

def load_model_weights(model, new_checkpoint):
    model.load_state_dict(torch.load(new_checkpoint))
    model.train() 
    return model

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, tokenizer, model , k_shot = -1, use_tagger = False, regressor = False):
        super(T5FineTuner, self).__init__()
        #self.log = logger
        self.model_name_or_path = hparams.model_name_or_path
        self.train_batch_size = hparams.train_batch_size
        self.eval_batch_size = hparams.eval_batch_size
        self.num_train_epochs = hparams.num_train_epochs
        self.learning_rate = hparams.learning_rate
        self.gradient_accumulation_steps = hparams.gradient_accumulation_steps
        self.weight_decay = hparams.weight_decay
        self.adam_epsilon = hparams.adam_epsilon
        self.warmup_steps = hparams.warmup_steps
        self.train_path = hparams.train_dataset_path
        self.dev_path = hparams.dev_dataset_path
        self.test_path = hparams.test_dataset_path
        self.task = hparams.task
        self.max_seq_length = hparams.max_seq_length
        self.n_gpu = hparams.n_gpu
        self.k_shot = k_shot
        self.use_tagger = use_tagger
        self.regressor = regressor

        ### model init
        self.tokenizer = tokenizer
        self.model = model

        ### result cache

        self.best_f1 =-999999.0
        self.best_checkpoint = "best_checkpoint_dev"
        self.best_epoch = None


        #### Tagger
        self.classifier = nn.Linear(768, 3)  ## 3 to 5 maybe 
        self.softmax = nn.Softmax(dim=2)
        self.tag_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.token_dropout = nn.Dropout(0.1)

        ### Regressor
        self.regressor_layer = nn.Linear(768,128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128,64)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(64,1)
        self.regressor_criterion = nn.MSELoss()



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
        print(loss, "loss before tag")
        if self.use_tagger:
            encoder_states = outputs.encoder_last_hidden_state  
            logits = self.classifier(self.token_dropout(encoder_states))
            tag_loss = self.tag_criterion(logits.view(-1, 3), batch['op_tags'].view(-1))  ## 3 to 5 maybe
            
            loss += tag_loss
            print(loss, "loss after tag")

        if self.regressor:
            encoder_states = outputs.encoder_last_hidden_state 
            mask_position = torch.tensor(np.where( batch["source_ids"].cpu().numpy() == 1, 1, 0)).to('cuda')
            masked_embeddings = encoder_states * mask_position.unsqueeze(2)

            sentence_embedding = torch.sum(masked_embeddings, axis = 1)
            normalized_sentence_embeddings = sentence_embedding.cuda()

            outs = self.regressor_layer(self.token_dropout(normalized_sentence_embeddings))
            outs = self.relu1(outs)
            outs = self.ff1(outs)
            outs = self.tanh1(outs)
            outs = self.ff2(outs)

            regressor_loss = self.regressor_criterion(outs, batch['triplet_count'].view(-1).type_as(outs))
            loss += 0.4 * regressor_loss  #### Hyperparameter 0.4
            print(loss, "loss after regression")
            
        
        return loss

    def _generate(self, batch):

        outs = self.model.generate(input_ids=batch['source_ids'].to('cuda'), 
                            attention_mask=batch['source_mask'].to('cuda'), 
                            max_length=128)
        outputs = []
        targets = []
        #print(outs)
        for i in range(len(outs)):

            dec = tokenizer.decode(outs[i], skip_special_tokens=False)
            labels = np.where(batch["target_ids"][i].cpu().numpy() != -100, batch["target_ids"][i].cpu().numpy(), tokenizer.pad_token_id)
            target = tokenizer.decode(torch.tensor(labels), skip_special_tokens=False)

            outputs.append(dec)
            targets.append(target)

        decoded_labels = correct_spaces(targets)
        decoded_preds = correct_spaces(outputs)
        print('decoded_preds', decoded_preds)
        print('decoded_labels', decoded_labels)

        linearized_triplets = {}
        linearized_triplets['predictions'] = decoded_preds
        linearized_triplets['labels'] = decoded_labels

        return linearized_triplets


    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        #logs = {"train_loss": loss}
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        print(outputs)
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss_after_epoch_end', avg_train_loss)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss)
        val_outs = {}
        val_outs['loss'] = loss
        generated_triplets = self._generate(batch)
        val_outs['predictions'] = generated_triplets['predictions']
        val_outs['labels'] = generated_triplets['labels']
        return val_outs


    def validation_epoch_end(self, outputs):
        
        custom_print('********************************************************************')
        custom_print('Epoch:', self.current_epoch)
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("avg_val_loss_after_epoch_end", avg_loss)
        all_preds = []
        all_labels = []
        #print(outputs)
        for i in range(len(outputs)):
            all_preds.extend(outputs[i]['predictions'])
            all_labels.extend(outputs[i]['labels'])

        p, r, f = get_f1_for_trainer(all_preds, all_labels )
        _, _, opinion_f = get_f1_for_trainer(all_preds, all_labels , 'opinion')
        _, _, aspect_f = get_f1_for_trainer(all_preds, all_labels , 'aspect')
        _, _, sentiment_f = get_f1_for_trainer(all_preds, all_labels , 'sentiment')

        if f > self.best_f1:
            self.best_f1 = f
            self.best_epoch = self.current_epoch
            torch.save(self.model.state_dict(), self.best_checkpoint)


        self.log('step', self.current_epoch)

        self.log('val f1', f, on_step=False, on_epoch=True)
        self.log('val prec', p, on_step=False, on_epoch=True)
        self.log('val rec', r, on_step=False, on_epoch=True)
        self.log('val opinion', opinion_f, on_step=False, on_epoch=True)
        self.log('val aspect', aspect_f, on_step=False, on_epoch=True)
        self.log('val sentiment', sentiment_f, on_step=False, on_epoch=True)

        custom_print('\nDev Results\n')
        custom_print('Dev Opinion F1:', round(opinion_f, 3))
        custom_print('Dev Aspect F1:', round(aspect_f, 3))
        custom_print('Dev Sentiment F1:', round(sentiment_f, 3))
        custom_print('Dev P:', round(p, 3))
        custom_print('Dev R:', round(r, 3))
        custom_print('Dev F1:', round(f, 3))
        
        return {'f1':f, 'prec': p, 'rec': r , 'opinion': opinion_f, 'aspect': aspect_f, 'sentiment': sentiment_f }

    def test_step(self, batch, batch_idx):

        test_outs = {}
        generated_triplets = self._generate(batch)
        test_outs['predictions'] = generated_triplets['predictions']
        test_outs['labels'] = generated_triplets['labels']
        return test_outs


    def test_epoch_end(self, outputs):
        
        all_preds = []
        all_labels = []
        #print(outputs)
        for i in range(len(outputs)):
            all_preds.extend(outputs[i]['predictions'])
            all_labels.extend(outputs[i]['labels'])

        p, r, f = get_f1_for_trainer(all_preds, all_labels )
        _, _, opinion_f = get_f1_for_trainer(all_preds, all_labels , 'opinion')
        _, _, aspect_f = get_f1_for_trainer(all_preds, all_labels , 'aspect')
        _, _, sentiment_f = get_f1_for_trainer(all_preds, all_labels , 'sentiment')

        self.log('step', self.current_epoch)
        self.log('test f1', f, on_step=False, on_epoch=True)
        self.log('test prec', p, on_step=False, on_epoch=True)
        self.log('test rec', r, on_step=False, on_epoch=True)
        self.log('test opinion', opinion_f, on_step=False, on_epoch=True)
        self.log('test aspect', aspect_f, on_step=False, on_epoch=True)
        self.log('test sentiment', sentiment_f, on_step=False, on_epoch=True)

        custom_print('\nTest Results\n')
        custom_print('Test Opinion F1:', round(opinion_f, 3))
        custom_print('Test Aspect F1:', round(aspect_f, 3))
        custom_print('Test Sentiment F1:', round(sentiment_f, 3))
        custom_print('Test P:', round(p, 3))
        custom_print('Test R:', round(r, 3))
        custom_print('Test F1:', round(f, 3))

        return {'f1':f, 'prec': p, 'rec': r , 'opinion': opinion_f, 'aspect': aspect_f, 'sentiment': sentiment_f }


    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                    epoch=None,
                    batch_idx=None,
                    optimizer=None,
                    optimizer_idx=None,
                    optimizer_closure=None,
                    on_tpu=None,
                    using_native_amp=None,
                    using_lbfgs=None):

                optimizer.step(closure=optimizer_closure)
                optimizer.zero_grad()
                self.lr_scheduler.step()
    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        
        

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_path =self.train_path, task = self.task, max_seq_length = self.max_seq_length, k_shot = self.k_shot )
        dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.train_batch_size * max(1, len(self.n_gpu))))
            // self.gradient_accumulation_steps
            * float(self.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        print("making val data")
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_path = self.dev_path, task = self.task, max_seq_length = self.max_seq_length )
        return DataLoader(val_dataset, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        print("making test data")
        test_dataset = get_dataset(tokenizer=self.tokenizer, data_path = self.test_path, task = self.task, max_seq_length = self.max_seq_length )
        return DataLoader(test_dataset, batch_size=self.eval_batch_size)


def evaluate(data_loader, model):

    #model.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        outs = model.model.generate(input_ids=batch['source_ids'].to('cuda'), 
                                    attention_mask=batch['source_mask'].to('cuda'), 
                                    max_length=128)
        for i in range(len(outs)):
            dec = tokenizer.decode(outs[i], skip_special_tokens=False)
            labels = np.where(batch["target_ids"][i] != -100, batch["target_ids"][i], tokenizer.pad_token_id)
            target = tokenizer.decode(labels, skip_special_tokens=False)

            outputs.append(dec)
            targets.append(target)


    decoded_labels = correct_spaces(targets)
    decoded_preds = correct_spaces(outputs)

    for l in decoded_preds:
        custom_print(l)
    #print(decoded_labels)

    p, r, f = get_f1_for_trainer(decoded_preds, decoded_labels )
    _, _, opinion_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'opinion')
    _, _, aspect_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'aspect')
    _, _, sentiment_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'sentiment')

    custom_print('\nTest Results\n')
    custom_print('Test Opinion F1:', round(opinion_f, 3))
    custom_print('Test Aspect F1:', round(aspect_f, 3))
    custom_print('Test Sentiment F1:', round(sentiment_f, 3))
    custom_print('Test P:', round(p, 3))
    custom_print('Test R:', round(r, 3))
    custom_print('Test F1:', round(f, 3))

    return {'f1':f, 'prec': p, 'rec': r , 'opinion': opinion_f, 'aspect': aspect_f, 'sentiment': sentiment_f }

if __name__ == '__main__':

    args = initialise_args()
    seed_everything(args.seed)

    sent_map = {}
    sent_map['POS'] = 'positive'
    sent_map['NEU'] = 'neutral'
    sent_map['NEG'] = 'negative'

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    
    tokenizer.add_tokens(['<triplet>', '<opinion>', '<sentiment>'], special_tokens = True)
    tuner_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tuner_model.resize_token_embeddings(len(tokenizer))
    tuner_model.to('cuda')

    if (args.model_weights != ''):  ## initializing checkpoint weights
        weights = args.model_weights
        tuner_model = load_model_weights(tuner_model, weights)

    #logger = logging.getLogger(__name__)
    #Replace with Custom WandB logger
    logger = TensorBoardLogger(args.output_dir, name='ASTE')
    custom_logger =  open(os.path.join(args.output_dir, args.logger_name), 'w')
    custom_print(args.log_message)

    k_shot = args.k_shot
    use_tagger = args.use_tagger
    regressor = args.regressor
    
    if args.do_train:

        custom_print("\n****** Conduct Training ******")

        
        model = T5FineTuner(args, tokenizer, tuner_model, k_shot, use_tagger, regressor )

        checkpoint_callback = []

        checkpoint_callback.append( ModelCheckpoint(
            monitor='val f1',
            # monitor=None,
            save_top_k=5,
            verbose=True,
            save_last=False,
            mode='max'
        ))


        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=checkpoint_callback,
        )

        
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        # model.model.save_pretrained(args.output_dir)

        custom_print("Finish training and saving the model!")

        custom_print("The best Dev epoch is:", model.best_epoch)


    
    if args.do_eval:

        custom_print("\n****** Conduct Evaluating ******")
        trainer.test(model)
        # model = T5FineTuner(args)
        dev_results, test_results = {}, {}
        best_f1, best_checkpoint, best_epoch = -999999.0, None, None
        all_checkpoints, all_epochs = [], []

        # saved_model_dir = args.output_dir
        # saved_model_dir = os.path.join(saved_model_dir, 'lightning_logs/version_1/checkpoints/')
        # for f in os.listdir(saved_model_dir):
        #     file_name = os.path.join(saved_model_dir, f)
        #     all_checkpoints.append(file_name)

        # print(f"We will perform validation on the following checkpoints: {all_checkpoints}")

        
            # load dev and test datasets
        dev_dataset = ASTE_Dataset(tokenizer, data_path=args.dev_dataset_path, task=args.task, max_len=args.max_seq_length)
        dev_loader = DataLoader(dev_dataset, batch_size=32)
        
        test_dataset = ASTE_Dataset(tokenizer, data_path=args.test_dataset_path, task=args.task, max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32)

        custom_print('*************Loading Checkpoint***************: ', model.best_checkpoint)
        model_ckpt = torch.load(model.best_checkpoint)
        eval_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        eval_model.resize_token_embeddings(len(tokenizer))
        eval_model.to('cuda')
        eval_model.load_state_dict(model_ckpt)
        tuner = T5FineTuner(args, tokenizer, eval_model)
        custom_print('**************** Printing Model Outputs for Test***************')
        _ = evaluate(test_loader, tuner)

        ## To DO:

        ## Correct the top performer all checkpoints loading below
        ## MAybe log all the generated sentences in test to the log file

        # for checkpoint in all_checkpoints:
        #     custom_print('****Loading Checkpoint***************: ', checkpoint)
        #     model_ckpt = torch.load(checkpoint)
        #     eval_model.load_state_dict(model_ckpt)
        #     tuner = T5FineTuner(args, tokenizer, eval_model)
        #     _ = evaluate(test_loader, tuner)




    custom_print("All Done :)")
    custom_logger.close()
    



