# -*- coding: utf-8 -*-
import os
import sys
import argparse
from easydict import EasyDict
import pprint
import yaml
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from modules.Utilities import Text_Classification_Document, TC_Dataset, TC_Predictor
from sklearn import metrics
import logging
import pandas as pd
import numpy as np
import re


def main():
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  logging.info('Text Classification training pipeline started')
  """ load config """
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config, encoding='utf-8') as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))
  
  logging.info('Config loaded:')
  pprint.pprint(config, sort_dicts=False)
  """ Load label_map """
  label_map = config['label_map']
  """ Load tokenizer """
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  """ make training datasets """
  logging.info('Creating datasets...')
  
  with open(config['deve_id_file']) as f:
    lines = f.readlines()
  dev_ids = [line.strip() for line in lines]
  # split into training and validation sets
  np.random.seed(123)
  valid_ids = np.random.choice(dev_ids, int(len(dev_ids) * config['valid_ratio']), 
                               replace=False).tolist()

  # Load training/ validation IEs into dict {doc_id, IE}
  valid_TCs = []
  for valid_id in valid_ids:
    valid_TCs.append(Text_Classification_Document(doc_id=valid_id, 
                                                  filename=os.path.join(config['TC_dir'], f'{valid_id}.tc')))
    
  valid_dataset = TC_Dataset(TCs=valid_TCs, 
                            tokenizer=tokenizer, 
                            sequence_length=config['sequence_length'], 
                            label_map=label_map, 
                            has_label=False)
      
  model = AutoModelForSequenceClassification.from_pretrained(config['base_model'], 
                                                             num_labels=len(label_map), 
                                                             problem_type="multi_label_classification" )
  
  if config['checkpoint'] == 'best':
    model_names = [f for f in os.listdir(os.path.join(config['out_path'], 'checkpoints', config['run_name'])) if '.pth' in f]
    best_model_name = sorted(model_names, key=lambda x:int(re.search("-(.*?)_", x).group(1)))[-1]
    logging.info(f'Evaluate model: {best_model_name}')
    model.load_state_dict(torch.load(os.path.join(config['out_path'], 'checkpoints', config['run_name'], best_model_name), 
                                    map_location=torch.device('cpu')))
  
  else:
    logging.info(f"Evaluate model: {config['checkpoint']}")
    model.load_state_dict(torch.load(os.path.join(config['out_path'], config['run_name'], config['checkpoint']), 
                                    map_location=torch.device('cpu')))
  logging.info('Model loaded')
  """ Prediction """
  logging.info('Predicting...')
  predictor = TC_Predictor(model=model,
                        tokenizer=tokenizer,
                        dataset=valid_dataset,
                        label_mode=config['label_mode'],
                        label_map=label_map,
                        batch_size=config['eval_batch_size'],
                        force_predict=False,
                        device=config['device'])
  
  pred_TCs = predictor.predict()  

  len(pred_TCs[0].predicted_prob)
  valid_TCs[0].label

  def search_cutoff(df:pd.DataFrame):
    if len(df['label'].unique()) < 2:
      return 0.5
    
    df.sort_values('prob', inplace=True)
    met = []
    for cut in df['prob'].unique():
      pred = df['prob'] > cut
      f1 = metrics.f1_score(df['label'], pred)
      met.append({'cutoff':cut, 'f1':f1})
    
    met = pd.DataFrame(met)
    return met.loc[met['f1'].idxmax()]['cutoff']

  
  cutoff_list = []
  for v in config['label_map'].keys():
    pred_tc_list = []
    for tc in pred_TCs:
      pred_tc_list.append({'doc_id':tc['doc_id'], 'prob':tc.predicted_prob[v]})
      
    gold_tc_list = []  
    for tc in valid_TCs:
      gold_tc_list.append({'doc_id':tc['doc_id'], 'label':v in tc.label})
      
    df = pd.merge(pd.DataFrame(pred_tc_list), pd.DataFrame(gold_tc_list), on='doc_id')
    cut = search_cutoff(df)
    cutoff_list.append({'level':v, 'cutoff':cut})
    
  cutoff = pd.DataFrame(cutoff_list)
  
  if not os.path.isdir(os.path.join(config['out_path'], 'cuf-off')):
    os.makedirs(os.path.join(config['out_path'], 'cuf-off'))
  cutoff.to_csv(os.path.join(config['out_path'], 'cuf-off', f"{config['run_name']}.csv"), index=False)
  
  
if __name__ == '__main__':
  main()