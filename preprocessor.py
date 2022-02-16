from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import os

def read_data(path):

    sents = open( path + '.sent', 'r')
    sentences = sents.readlines()
    tups = open(path +  '.tup', 'r')
    tuples = tups.readlines()

    return sentences, tuples

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


def get_transformed_data(sentences_list, tuples_list):
    """
    Preprocess the raw data into Generative Targets
    """
    inputs = []
    targets = []
    
    for i in range(len(sentences_list)):
        
        sent = sentences_list[i].strip()
        tup = tuples_list[i]
        tup_dict = generate_triplet_dict(tup)
        target = generate_target(tup_dict)
        inputs.append(sent)
        targets.append(target)

    return inputs, targets

class ASTE_Dataset(Dataset):
    def __init__(self, tokenizer, data_path , task, max_len=128):
        # 'data/aste/rest16/train.txt'
        self.data_path = data_path
        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()      # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        sentences, tuples = read_data(self.data_path)
        inputs, targets = get_transformed_data(sentences, tuples)

        for i in range(len(inputs)):

            input = inputs[i]
            target = targets[i]

            tokenized_input = self.tokenizer(
              [input], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt",
            )
            with self.tokenizer.as_target_tokenizer():
                tokenized_target = self.tokenizer(
                [target], max_length=self.max_len, pad_to_max_length=True, truncation=True,
                return_tensors="pt"
                )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)