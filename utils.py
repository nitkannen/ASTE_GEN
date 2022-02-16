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
