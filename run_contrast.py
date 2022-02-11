
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch import nn
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
import datasets
import random
import sentencepiece

from IPython.display import display, HTML
import nltk
nltk.download('punkt')

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
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_metric

#raw_datasets = load_dataset("xsum")
metric = load_metric("rouge")

from collections import OrderedDict

### Train

sents = open('15res/train.sent', 'r')
train_sentences = sents.readlines()
tups = open('15res/train.tup', 'r')
train_tuples = tups.readlines()

### Dev

sents = open('15res/dev.sent', 'r')
dev_sentences = sents.readlines()
tups = open('15res/dev.tup', 'r')
dev_tuples = tups.readlines()

## test

sents = open('15res/test.sent', 'r')
test_sentences = sents.readlines()
tups = open('15res/test.tup', 'r')
test_tuples = tups.readlines()

def generate_triplet_dict(tuples):
  """
  takes a set of tuples and generates triplet dictionary
  """
  triplets = tuples.split('|')
  d = OrderedDict()
  for triplet in triplets:
    a, o, s = triplet.split(';')
    if(a.strip() in d.keys()):
      d[a.strip()].append((o.strip(), s.strip()))
    else:
      d[a.strip()] = []
      d[a.strip()].append((o.strip(), s.strip()))
  
  return d  



sent_map = {}
sent_map['POS'] = 'positive'
sent_map['NEU'] = 'neutral'
sent_map['NEG'] = 'negative'

def generate_target(d):
  """
  takes a aspect triple dictionary and linearizes it
  """
  summary = ""
  if len(d.items()) == 0:
    return summary
  for items in d.items():
    summary += '<triplet> '
    summary += items[0] + ' '
    for opinion in items[1]:
      summary += '<opinion> '
      summary += opinion[0] + ' '
      summary += '<sentiment> '
      summary += sent_map[opinion[1]] + ' '

  return summary.strip()


def create_df(sentences_list, tuples_list):
  examples = []
  for i in range(len(sentences_list)):
    ex = {}
    sent = sentences_list[i].strip()
    tup = tuples_list[i]
    tup_dict = generate_triplet_dict(tup)
    target = generate_target(tup_dict)
    ex['input'] = sent
    ex['summary'] = target

    examples.append(ex)

  
  data_df = pd.DataFrame(examples)

  return data_df

train_df = create_df(train_sentences, train_tuples)
dev_df = create_df(dev_sentences, dev_tuples)
test_df = create_df(test_sentences, test_tuples)

train_df.to_csv('train.csv', index = False)
dev_df.to_csv('val.csv', index = False)
test_df.to_csv('test.csv', index = False)
raw_datasets = load_dataset('csv', data_files={ 'train':'train.csv', 'validation': 'val.csv', 'test': 'test.csv'})

from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained('t5-base')
tokenizer.add_tokens(['<triplet>', '<opinion>', '<sentiment>'], special_tokens = True)

max_input_length = 128
max_target_length = 128
prefix = ""

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    del(examples['summary'])
    del(examples['input'])
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)





def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

    decoded_labels = correct_spaces(decoded_labels)
    #print(decoded_labels)
    decoded_preds = correct_spaces(decoded_preds)
    p, r, f = get_f1_for_trainer(decoded_preds, decoded_labels )
    _, _, opinion_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'opinion')
    _, _, aspect_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'aspect')
    _, _, sentiment_f = get_f1_for_trainer(decoded_preds, decoded_labels , 'sentiment')


    return {'F1':f, 'Prec': p, 'Rec': r , 'Opinion': opinion_f, 'Aspect': aspect_f, 'Sentiment': sentiment_f }


def get_f1_for_trainer(predictions, target, component = None):


  n = len(target)
  assert n == len(predictions)

  preds, gold = [], []
  
  for i in range(n):
    
    preds.append(decode_pred_triplets(predictions[i]))
    gold.append(decode_pred_triplets(target[i]))

  pred_triplets = 0
  gold_triplets = 0
  correct_triplets = 0

  for i in range(n):

    pred_triplets += len(preds[i])
    gold_triplets += len(gold[i])

    for gt_triplet in gold[i]:

      if component is None and is_full_match(gt_triplet, preds[i]):
        correct_triplets += 1
      elif component is 'aspect' and is_full_match(gt_triplet, preds[i], aspect = True):
        correct_triplets += 1
      elif component is 'opinion' and is_full_match(gt_triplet, preds[i], opinion = True):
        correct_triplets += 1
      elif component is 'sentiment' and is_full_match(gt_triplet, preds[i], sentiment = True):
        correct_triplets += 1
    


  p = float(correct_triplets) / (pred_triplets + 1e-8 )
  r = float(correct_triplets) / (gold_triplets + 1e-8 )
  f1 = (2 * p * r) / (p + r + 1e-8)

  return p, r, f1


def correct_spaces(result):

    for i in range(len(result)):
        s = ''
        for char in result[i]:
            if char == '<':
                s += ' ' + char
            else:
                s += char

        result[i] = s

    return result

def post_process(text):
  if len(text) > 9:
    if text[:9] != '<triplet>':
      text = '<triplet>' + text
  return text


""" adapted from https://github.com/Babelscape/rebel/blob/main/src/utils.py"""

def decode_pred_triplets(text):

  triplets = []
  text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
  text_processed = post_process(text)
  current = None
  aspect, opinion, sentiment = "", "", ""
  #?print(text_processed)
  for token in text_processed.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
    #print(token)
    if token == '<triplet>':
      current = 't'
      if sentiment != "":
        triplets.append({"aspect": aspect.strip(), "opinion": opinion.strip(), "sentiment" : sentiment.strip()})
        sentiment = ""
      aspect = ""

    elif token == '<opinion>':

      current = 'o'
      if sentiment != "":
        triplets.append({"aspect": aspect.strip(), "opinion": opinion.strip(), "sentiment" : sentiment.strip()})
      opinion = ""

    elif token == '<sentiment>':
      current = 's'
      sentiment = ""

    else:
      if current == 't':
        aspect += ' ' + token
      elif current == 'o':
        opinion += ' ' + token
      elif current =='s':
        sentiment += ' ' + token

  if aspect != '' and opinion != '' and sentiment != '':
    triplets.append({"aspect": aspect.strip(), "opinion": opinion.strip(), "sentiment" : sentiment.strip()})

  return triplets




def get_gold_triplets(dev_target_sample):

  triplets = dev_target_sample.split('|')
  triplets_list = []
  for triplet in triplets:

    d = {}
    a, o, s = triplet.split(';')
    d['aspect'] = a.strip()
    d['opinion'] = o.strip()
    d['sentiment'] = sent_map[s.strip()]
    triplets_list.append(d)

  return triplets_list

def is_full_match(triplet, triplets, aspect = None, opinion = None, sentiment = None):



  for t in triplets:

    if aspect:
      if t['aspect'] == triplet["aspect"]:
          return True;
    elif opinion:
      if t['opinion'] == triplet['opinion']:
          return True;
    elif sentiment:
      if t['sentiment'] == triplet['sentiment']:
          return True;
    else:
      if t['opinion'] == triplet['opinion'] and t['aspect'] == triplet["aspect"] and t['sentiment'] == triplet['sentiment']:
          return True

  return False


def get_f1(predictions, target, component = None):


  n = len(target)
  assert n == len(predictions)

  preds, gold = [], []

  for i in range(n):
    preds.append(decode_pred_triplets(predictions[i]))
    gold.append(get_gold_triplets(target[i]))


  pred_triplets = 0
  gold_triplets = 0
  correct_triplets = 0

  for i in range(n):

    pred_triplets += len(preds[i])
    gold_triplets += len(gold[i])

    for gt_triplet in gold[i]:

      if component is None and is_full_match(gt_triplet, preds[i]):
        correct_triplets += 1
      elif component is 'aspect' and is_full_match(gt_triplet, preds[i], aspect = True):
        correct_triplets += 1
      elif component is 'opinion' and is_full_match(gt_triplet, preds[i], opinion = True):
        correct_triplets += 1
      elif component is 'sentiment' and is_full_match(gt_triplet, preds[i], sentiment = True):
        correct_triplets += 1
    
    


  p = float(correct_triplets) / (pred_triplets + 1e-8 )
  r = float(correct_triplets) / (gold_triplets + 1e-8 )
  f1 = (2 * p * r) / (p + r + 1e-8)

  return p, r, f1


def load_model_weights(model, new_checkpoint):
    model.load_state_dict(torch.load(new_checkpoint))
    model.train()
    return model

def initialize_model_trainer(model = None, weights = None):

  if model is None:
    from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    model.resize_token_embeddings(len(tokenizer))
    model.to('cuda')

  if weights is not None:
    model = load_model_weights(model, weights)


  batch_size = 2
  model_name = 't5 for aste'#model_checkpoint.split("/")[-1]
  args = Seq2SeqTrainingArguments(
      model_name,
      evaluation_strategy = "epoch",
      learning_rate=3e-4,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      weight_decay=0.01,
      save_total_limit=3,
      num_train_epochs=20,
      predict_with_generate=True,
  )


  
  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding = True)

  trainer = Seq2SeqTrainer(
      model,
      args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["test"],
      data_collator=data_collator,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics
      )

  return trainer, model



class ABSADataset(Dataset):
    def __init__(self, data_samples, tokenizer):
        super(ABSADataset, self).__init__()
        self.tokenizer = tokenizer
        self.t5_input = [self.tokenizer(data_samples[i]['sentence']) for i in range(len(data_samples) )]
        self.t5_input_ids = [self.t5_input[i]['input_ids'] for i in range(len(data_samples))]
        self.t5_attention_mask = [self.t5_input[i]['attention_mask'] for i in range(len(data_samples))]
        self.raw_texts = [data_samples[i]['sentence'] for i in range(len(data_samples))]
        
        self.labels = [data_samples[i]['label'] for i in range(len(data_samples))]
        self.len = len(data_samples)

    def __getitem__(self, index):
        return (self.t5_input_ids[index],
                self.t5_attention_mask[index],
                self.labels[index],
                self.raw_texts[index])

    def __len__(self):
        return self.len

def collate_fn(batch):


    input_ids, attention_mask, labels, raw_text = zip(*batch)

    #bert_masks = pad_sequence([torch.ones(tokens.shape) for tokens in bert_tokens], batch_first=True)
    #bert_tokens = pad_sequence(bert_tokens, batch_first=True)
    #aspect_masks = pad_sequence(aspect_masks, batch_first=True)
    #input_ids = torch.tensor(input_ids)
    input_ids = pad_sequence([torch.tensor(input) for input in (input_ids)], batch_first=True)
    attention_mask = pad_sequence([torch.tensor(att) for att in (attention_mask)], batch_first=True)
    labels = torch.tensor(labels)
    #labels = torch.stack(labels)
  

    return (input_ids, attention_mask, labels)

class SupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-30)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def get_optimizer_grouped_parameters(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters

def get_optimizer_scheduler(model, train_dataloader, epochs):

    total_steps = len(train_dataloader) * epochs
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

    return optimizer, scheduler

def has_opposite_labels(labels):
    return not (labels.sum().item() <= 1 or (1 - labels).sum().item() <= 1)

f = open('Contrastive/Res_plus_Lap_contrast.json')

contrast_examples =  json.load(f)
Contrastive_Dataset = ABSADataset(contrast_examples, tokenizer)
loader = DataLoader(
                    Contrastive_Dataset, collate_fn = collate_fn ,shuffle = True,  batch_size=16
          )

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
model.resize_token_embeddings(len(tokenizer))

epochs = 10
current_step = 0
contrast_criterion = SupConLoss()
optimizer, scheduler = get_optimizer_scheduler(model, loader, epochs)
trainer_model = None

################## Conttrastive Training clubbed with the conventional training

for epoch in range(epochs):

  start = time.time()
        
  total_loss = 0

  if(epoch % 2 == 0):
      print('/////_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_/////')
      print("Training AFter: Epochs: ", epoch)
      trainer, trainer_model = initialize_model_trainer(trainer_model, model)
      trainer.train()
  
  ## Possible experiment
  # loader = DataLoader(
  #                   Contrastive_Dataset, collate_fn = collate_fn ,shuffle = True,  batch_size=16
  #         )
  for idx, batch in enumerate(loader):
      
          
        model.train()
        
        
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        labels = labels.to('cuda')

        if has_opposite_labels(labels):
          outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              decoder_input_ids = input_ids
          )
          masked_encoder_embeddings = outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2)
          average_encoder_embeds = torch.sum(masked_encoder_embeddings, axis = 1) / attention_mask.unsqueeze(2).sum(axis = 1)

          normalized_sentence_embeddings = average_encoder_embeds  ## populate this  ## encoder average, decoder average, fit heads maybe
          normalized_sentence_embeddings = normalized_sentence_embeddings.cuda()
          similar_loss = contrast_criterion(normalized_sentence_embeddings.unsqueeze(1), labels=labels)
          loss = similar_loss

          optimizer.zero_grad()
          loss.backward()
          total_loss += loss.item()

          optimizer.step()
          scheduler.step()

          current_step += 1

          #print("Loss in Batch:", loss.item())

        else:
            pass
       
  print("#########################################################################")
  print("Loss after Epoch:", total_loss / 16)

  end = time.time()
  print("[Epoch {:2d}] complete in {:.2f} seconds".format(epoch, end - start))