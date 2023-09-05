# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import pprint
import yaml
import os
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch.optim as optim
from modules.Utilities import Text_Classification_Document, TC_Dataset, TC_Trainer
import logging
import numpy as np

def main():
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  logging.info('Text Classification training pipeline started')
  """ load config """
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
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
  train_ids = [i for i in dev_ids if i not in valid_ids]
  # Load training/ validation IEs into dict {doc_id, IE}
  train_TCs = []
  for train_id in train_ids:
    train_TCs.append(Text_Classification_Document(doc_id=train_id, 
                                                  filename=os.path.join(config['TC_dir'], f'{train_id}.tc')))
      
  valid_TCs = []
  for valid_id in valid_ids:
    valid_TCs.append(Text_Classification_Document(doc_id=valid_id, 
                                                  filename=os.path.join(config['TC_dir'], f'{valid_id}.tc')))
      
  train_dataset = TC_Dataset(TCs=train_TCs, 
                             tokenizer=tokenizer, 
                             sequence_length=config['sequence_length'], 
                             label_map=label_map, 
                             has_label=True)
  
  valid_dataset = TC_Dataset(TCs=valid_TCs, 
                             tokenizer=tokenizer, 
                             sequence_length=config['sequence_length'], 
                             label_map=label_map, 
                             has_label=True)

  logging.info('Datasets created')
  
  """ define model """
  logging.info(f"Loading base model from {config['base_model']}...")
  if train_dataset.label_mode == 'single-label':
    mode = "single_label_classification" 
  elif train_dataset.label_mode == 'multi-label':
    mode = "multi_label_classification" 
  
  model = AutoModelForSequenceClassification.from_pretrained(config['base_model'], 
                                                             num_labels=len(label_map),
                                                             problem_type=mode)
  logging.info('Model loaded')
  """ Load optimizer """
  optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
  """ Training """
  trainer = TC_Trainer(run_name=config['run_name'], 
                        model=model,
                        n_epochs=config['n_epochs'],
                        train_dataset=train_dataset,
                        batch_size=config['batch_size'],
                        optimizer=optimizer,
                        valid_dataset=valid_dataset,
                        save_model_mode='best',
                        save_model_path=os.path.join(config['out_path'], 'checkpoints'),
                        log_path=os.path.join(config['out_path'], 'logs'),
                        device=config['device'])
  
  trainer.train()

if __name__ == '__main__':
  main()