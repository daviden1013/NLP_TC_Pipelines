# -*- coding: utf-8 -*-
import os
import re
import argparse
from easydict import EasyDict
import yaml
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from modules.Utilities import Text_Classification_Document, TC_Dataset, TC_Predictor, TC_Evaluator
import logging
import pprint


def main():
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  logging.info('Evaluation pipeline starts:')
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))

  logging.info('Config loaded:')
  pprint.pprint(config, sort_dicts=False)
  """ Load label map """
  label_map = config['label_map']
  """ load tokenizer """
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  """ load test_id """
  logging.info('Creating datasets...')
  with open(config['test_id_file']) as f:
    lines = f.readlines()
  test_ids = [line.strip() for line in lines]

  test_TCs = []
  for test_id in test_ids:
    test_TCs.append(Text_Classification_Document(doc_id=test_id, 
                                                 filename=os.path.join(config['TC_dir'], f'{test_id}.tc')))
      
  test_dataset = TC_Dataset(TCs=test_TCs, 
                            tokenizer=tokenizer, 
                            sequence_length=config['sequence_length'], 
                            label_map=label_map, 
                            has_label=False)
  logging.info('Datasets created')
  """ load model """
  logging.info('Loading model...')
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
                        dataset=test_dataset,
                        label_mode=config['label_mode'],
                        label_map=label_map,
                        batch_size=config['eval_batch_size'],
                        device=config['device'])
  
  pred_TCs = predictor.predict()  
  logging.info('TCs predicted')
  """ Evaluate """

  logging.info('Evaluating...')
  evaluator = TC_Evaluator(pred_TCs, test_TCs, config['label_mode'])
  total_eval, doc_eval = evaluator.evaluate()
  # Make evaluation output directory
  if not os.path.isdir(os.path.join(config['out_path'], 'evaluation', config['run_name'])):
      os.makedirs(os.path.join(config['out_path'], 'evaluation', config['run_name']))
  
  total_eval.to_csv(os.path.join(config['out_path'], 'evaluation', config['run_name'], 'Overall.csv'),
               index=False)
  doc_eval.to_csv(os.path.join(config['out_path'], 'evaluation', config['run_name'], 'document.csv'),
               index=False)

if __name__ == '__main__':
  main()
