# -*- coding: utf-8 -*-
PATH = 
import os
import numpy as np

files = [f.replace('.tc', '') for f in os.listdir(os.path.join(PATH, 'TC'))]
def train_test_split(id_list, test_ratio=0.2):
  np.random.seed(123)
  test_id = np.random.choice(id_list, int(len(id_list) * test_ratio), replace=False).tolist()
  train_id = [i for i in id_list if i not in test_id]
  return train_id, test_id

train_id, test_id = train_test_split(files)

with open(os.path.join(PATH, 'doc_id', 'train_id'), 'w') as f:
  for l in train_id:
    f.write("%s\n" % l)
    
with open(os.path.join(PATH, 'doc_id', 'test_id'), 'w') as f:
  for l in test_id:
    f.write("%s\n" % l)