# -*- coding: utf-8 -*-
from typing import List, Union
import argparse
from easydict import EasyDict
import yaml
from modules.Utilities import Label_studio_TC_converter
import logging
import pprint
  

class My_TC_converter(Label_studio_TC_converter):
  def __init__(self, 
               doc_id_var:str, 
               ann_file:str, 
               TC_dir:str):
    super().__init__(doc_id_var, ann_file, TC_dir)
    
    
  def _parse_label(self, idx:int) -> Union[str, List[str]]:
    ann = self.annotation[idx]
    label = []
    for r in ann['annotations'][0]['result']:
      if r['type'] == 'choices':
        label.append(f"{r['from_name']}{r['value']['choices'][0]}")

    return label
  

def main():
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  logging.info('TC pipeline started')
  
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))
    
  logging.info('Config loaded:')
  pprint.pprint(config, sort_dicts=False)
  
  logging.info('Converting...')
  converter = My_TC_converter(doc_id_var=config['doc_id_var'],
                              ann_file=config['ann_file'],
                              TC_dir=config['TC_dir'])
  
  converter.pop_TC()
  logging.info('TC pipeline finished')

if __name__ == '__main__':
  main()