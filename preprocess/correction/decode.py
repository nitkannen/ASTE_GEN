#from utils import correct_spaces, get_f1_for_trainer, decode_pred_triplets, is_full_match
from collections import OrderedDict

sent_map = {}
sent_map['POS'] = 'positive'
sent_map['NEU'] = 'neutral'
sent_map['NEG'] = 'negative'

def post_process(text):
    if len(text) > 9:
        if text[:9] != '<triplet>':
            text = '<triplet>' + text
    return text

def decode_pred_triplets(text):

  triplets = []
  text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("<extra_id_-1>", '<triplet>').replace("<extra_id_-2>", '<opinion>' ).replace("<extra_id_-3>", '<sentiment>')
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

def get_f1_for_trainer(predictions, target, component = None):
    
  # print(predictions)
  # print(target)

  n = len(target)
  assert n == len(predictions)

  preds, gold = [], []
  
  for i in range(n):
    
    preds.append(decode_pred_triplets(predictions[i]))
    gold.append(decode_pred_triplets(target[i]))
    #print(decode_pred_triplets(predictions[i]))
    #print(target[i])

  pred_triplets = 0
  gold_triplets = 0
  correct_triplets = 0

  for i in range(n):

    pred_triplets += len(preds[i])
    gold_triplets += len(gold[i])

    for gt_triplet in gold[i]:

      if component is None and is_full_match(gt_triplet, preds[i]):
        correct_triplets += 1
      
    


  p = float(correct_triplets) / (pred_triplets + 1e-8 )
  r = float(correct_triplets) / (gold_triplets + 1e-8 )
  f1 = (2 * p * r) / (p + r + 1e-8)

  return p, r, f1
  

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

def generate_triplet_dict(tuples, sentence):
    """
    takes a set of tuples and generates triplet dictionary
    """
    triplets = tuples.split('|')
    d = OrderedDict()
    ordered_triplets = []
    for triplet in triplets:
        #print(triplet)
        a, o, _ = triplet.split(';')
        ordered_triplets.append( ( sentence.find(a.strip()), sentence.find(o.strip())  , triplet) )
    #print(ordered_triplets)
    ordered_triplets = sorted(ordered_triplets)
    #print(ordered_triplets)
    for triplet in ordered_triplets:
        a, o, s = triplet[2].split(';')
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
        tup_dict = generate_triplet_dict(tup, sent)
        target = generate_target(tup_dict)
        inputs.append(sent)
        targets.append(target)

    return inputs, targets

if __name__ == '__main__':


    filename = '15res_contraste.txt'
    output_filename = 'decoded_triplets.txt'
    sents = open( filename, 'r', encoding = 'utf8')
    pred_tups = sents.readlines()
    #print(sents[0])
    pred_triplets =  []

    with open(output_filename, 'w') as f:
        for i in range(len(pred_tups)):
            pred_triplets.append(decode_pred_triplets(pred_tups[i]))
            tri = decode_pred_triplets(pred_tups[i])
            s = ''
            for t in tri:
                s += ' ' + t['aspect'] + ' ;'
                s += ' ' + t['opinion'] + ' ;'
                s += ' ' + t['sentiment'] + ' |'
            f.write(s)
            f.write('\n')

    sents = open( 'test.sent', 'r')
    sentences = sents.readlines()
    tups = open('test.tup', 'r')
    tuples = tups.readlines()

    _, target = get_transformed_data(sentences, tuples)

    p, r, f = get_f1_for_trainer( pred_tups, target )
    print(round(p, 3))
    print(round(r, 3))
    print(round(f, 3))