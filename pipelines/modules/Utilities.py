# -*- coding: utf-8 -*-
import abc
from typing import List, Dict, Tuple, Union
import os
from tqdm import tqdm
import yaml
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn import metrics
import warnings


class Text_Classification_Document:
  def __init__(self, 
               doc_id:str,
               filename:str=None,
               text:str=None, 
               label:Union[str, List[str]]=None,
               predicted_prob:Dict[str,float]=None):
    """
    Text Classification Document (TC) is a general object for single-label or 
    multi-label text classification
    
    Parameters
    ----------
    doc_id : str
      Document ID. Must be a string.
    filename : str, Optional
      file (.tc) path to load. If provided, the text, entity_list, relation_list
      parameters will not be used. 
    text : str, Optional
      document text.
    label : Union[str, List[str]]
      str (single-label) or list of str (multi-label)
    predicted_prob : Dict[str,float]
      dict of model-predicted probabilities. 
      If single-label, the probs sum to 1.0
      if multi-label, each prepresents the prob of having the label.
    """
    assert filename or text, "A filename or a text must be provided."
    assert isinstance(doc_id, str), "doc_id must be a string."
    self.doc_id = doc_id
    # if create object from file
    if filename:
      with open(filename) as yaml_file:
        tc = yaml.safe_load(yaml_file)
      if 'text' in tc.keys():
        self.text = tc['text']
      if 'label' in tc.keys():
        self.label = tc['label']
      if 'predicted_prob' in tc.keys():
        self.predicted_prob = tc['predicted_prob']
    # create object from raw inputs
    else:
      self.text = text
      if label is not None: 
        self.label = label
      if predicted_prob is not None:
        self.predicted_prob = predicted_prob
      
    # Assign label model
    if self.has_label():
      if isinstance(self.label, str):
        self.label_mode = 'single-label'
      
      elif isinstance(self.label, List):
        self.label_mode = 'multi-label'
      
    
  def __getitem__(self, key):
    if key in ["doc_id", "text", "label", "predicted_prob"]:
      return getattr(self, key)
    else:
      raise KeyError(f"'{key}' is not a valid key.")
    
  def has_label(self) -> bool:
    return hasattr(self, 'label')
  
  def has_predicted_prob(self) -> bool:
    return hasattr(self, 'predicted_prob')
    
  
  def __repr__(self, N_top_chars:int=100, N_top_items:int=5) -> str:
    text_to_print = self.text[0:N_top_chars]
    
    return (f'Information_Extraction_Document(doc_id="{self.doc_id}")\n',
            f'text="{text_to_print}", \n',
            f'label={self.label}, \n')

  def save(self, filename:str):
    with open(filename, 'w') as yaml_file:
      tc_dict = {'doc_id':self.doc_id, 'text':self.text}
      if self.has_label():
        tc_dict['label'] = self.label
      if self.has_predicted_prob():
        tc_dict['has_predicted_prob'] = self.predicted_prob
        
      yaml.safe_dump(tc_dict, yaml_file, sort_keys=False)
      yaml_file.flush()
      
  
class TC_converter:
  def __init__(self, TC_dir:str):
    """
    This class inputs a directory with annotation files, outputs TCs

    Parameters
    ----------
    ann_dir : str
      Directory of annotation files
    TC_dir : str
      Directory of TC files 
    """
    self.TC_dir = TC_dir

  @abc.abstractmethod
  def _parse_text(self) -> str:
    """
    This method inputs annotation filename with dir
    outputs the text
    """
    return NotImplemented

  @abc.abstractmethod
  def _parse_label(self) -> List[Dict[str, str]]:
    """
    This method inputs annotation filename with dir
    if single-label, outputs a str 
    if multi-label, outputs a dict {label_name, label}
    """
    return NotImplemented
  
  
  @abc.abstractmethod
  def pop_TC(self):
    """
    This method populates input annotation files and save as [doc_id].tc files. 
    """
    return NotImplemented
  

class Label_studio_TC_converter(TC_converter):
  def __init__(self, 
               doc_id_var:str, 
               ann_file:str, 
               TC_dir:str):
    """
    This class inputs an annotation file, outputs TCs
    Note that self._parse_label() remains undefined. Each project needs to inherit
    this class and define their own self._parse_label(). This is because Label-studio
    allows users to define their label collection structure. So the output for each
    project varies.
    
    Parameters
    ----------
    txt_dir : str
      Directory of text files
    ann_dir : str
      Directory of annotation files
    TC_dir : str
      Directory of TC files 
    """
    self.doc_id_var = doc_id_var
    self.ann_file = ann_file
    with open(self.ann_file, encoding='utf-8') as f:
      self.annotation = json.loads(f.read())
      
    self.TC_dir = TC_dir
  
  
  def _parse_doc_id(self, idx:int) -> str:
    ann = self.annotation[idx]
    return str(ann['data'][self.doc_id_var])
  
  
  def _parse_text(self, idx:int) -> str:
    ann = self.annotation[idx]
    return ann['data']['text']
  
  
  def pop_TC(self):
    """
    This method iterate through annotation files and create TCs
    """
    loop = tqdm(range(len(self.annotation)), total=len(self.annotation), leave=True)
    for i in loop:
      doc_id = self._parse_doc_id(i)
      txt = self._parse_text(i)
      label = self._parse_label(i)
      tc = Text_Classification_Document(doc_id=doc_id, 
                                        text=txt, 
                                        label=label)
      
      tc.save(os.path.join(self.TC_dir, f'{doc_id}.tc'))
      
      
class TC_Dataset(Dataset):
  def __init__(self, 
               TCs: List[Text_Classification_Document], 
               tokenizer: AutoTokenizer, 
               sequence_length: int,
               label_map: Dict[str, int],
               has_label: bool=True):
    """
    This parent class inputs list of TCs and for any combination of 2 entities, 
    outputs the segment of tokens (tokenizer) as dict 
    {document_id, input_ids, attention_mask, (labels)}
    number of tokens per input is padded/ truncated to token_length

    Parameters
    ----------
    TCs : List[Text_Classification_Document]
      List of TCs.
    tokenizer : AutoTokenizer
      tokenizer.
    sequence_length : int
      The number of tokens to input to model. Truncation and padding will be applied.
    label_map : Dict[str, int]
      a dict of {label:id}
    has_label : bool, optional
      Indicates if the TC has label. The default is True.
    """
    self.TCs = TCs
    self.tokenizer = tokenizer
    self.sequence_length = sequence_length
    self.label_map = label_map
    self.has_label = has_label
    # get label mode (single or multi-label)
    if self.has_label:
      label_mode = {tc.label_mode for tc in self.TCs}
      assert len(label_mode) == 1, "All TCs are expected to have same label_mode."
      self.label_mode = label_mode.pop()
    
  
  def __len__(self) -> int:
    return len(self.TCs)
  
  def __getitem__(self, idx:int) -> Dict:
    tc = self.TCs[idx]
    
    tokens = self.tokenizer(tc['text'], 
                            padding='max_length',
                            max_length=self.sequence_length,
                            truncation=True,
                            return_token_type_ids=False)
    
    tokens['doc_id'] = tc['doc_id']
    tokens['input_ids'] = torch.tensor(tokens['input_ids'])
    tokens['attention_mask'] = torch.tensor(tokens['attention_mask'])
    if self.has_label:
      if self.label_mode == 'single-label':
        tokens['label'] = torch.tensor(self.label_map[tc['label']])
    
      elif self.label_mode == 'multi-label':
        dim = len(self.label_map)       
        label_ids = [self.label_map[l] for l in tc['label']]
        tokens['label'] = torch.zeros(dim)
        tokens['label'][label_ids] = 1
          
    return tokens
    

class TC_Trainer():
  def __init__(self, 
               run_name: str, 
               model: AutoModelForSequenceClassification, 
               n_epochs: int, 
               train_dataset: Dataset, 
               batch_size: int, 
               optimizer, 
               valid_dataset: Dataset=None, 
               shuffle: bool=True, 
               drop_last: bool=True,
               save_model_mode: str="best", 
               save_model_path: str=None, 
               early_stop: bool=True,
               early_stop_epochs: int=8,            
               log_path: str=None, 
               device:str=None):    
    """
    This class trains a model with taining dataset and outputs checkpoints.

    Parameters
    ----------
    run_name : str
      Name of the run/ experiment.
    model : Union[AutoModelForTokenClassification, AutoModelForSequenceClassification]
      A base model to train.
    n_epochs : int
      Number of epochs.
    train_dataset : Dataset
      Training dataset.
    batch_size : int
      Batch size.
    optimizer : TYPE
      optimizer.
    valid_dataset : Dataset, optional
      Validation dataset. The default is None.
    shuffle : bool, optional
      Indicator for shuffling training instances. The default is True.
    drop_last : bool, optional
      Drop last training batch, so the shape of each batch is same. The default is True.
    save_model_mode : str, optional
      Must be one of {"best", "all"}. The default is best.
    save_model_path : str, optional
      Path for saving checkpoints. The default is None.
    early_stop : bool, optional
      Indicator for early stop
    early_stop_epochs : int, optional
      if early_stop=True, a continuous n epoches that the validation loss does not 
      drop will result in early stop.
    log_path : str, optional
      Path for saving logs. The default is None.
    device : str, optional
      CUDA device name. The default is cuda:0 if available, or cpu.
    """
    if device:
      self.device = device
    else: 
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      
    self.run_name = run_name
    self.model = model
    self.model.to(self.device)
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.valid_dataset = valid_dataset
    self.shuffle = shuffle
    self.save_model_mode = save_model_mode
    self.save_model_path = os.path.join(save_model_path, self.run_name)
    if save_model_path != None and not os.path.isdir(self.save_model_path):
      os.makedirs(self.save_model_path)
    self.best_loss = float('inf')
    self.early_stop = early_stop
    self.loss_no_drop_epochs = 0
    self.early_stop_epochs = early_stop_epochs
    self.train_dataset = train_dataset
    self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                   shuffle=self.shuffle, drop_last=drop_last)
    if valid_dataset != None:
      self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, 
                                     shuffle=False, drop_last=drop_last)
    else:
      self.valid_loader = None
    
    self.log_path = os.path.join(log_path, self.run_name)
    if log_path != None and not os.path.isdir(self.log_path):
      os.makedirs(self.log_path)
    self.tensorboard_writer = SummaryWriter(self.log_path) if log_path != None else None
    self.global_step = 0
    
  
  def save_model(self, epoch, train_loss, valid_loss):
    torch.save(self.model.state_dict(), 
               os.path.join(self.save_model_path, 
                            f'Epoch-{epoch}_trainloss-{train_loss:.4f}_validloss-{valid_loss:.4f}.pth'))
  
    
  def evaluate(self):
    with torch.no_grad():
      valid_total_loss = 0
      for valid_batch in self.valid_loader:
        valid_input_ids = valid_batch['input_ids'].to(self.device)
        valid_attention_mask = valid_batch['attention_mask'].to(self.device)
        valid_labels = valid_batch['label'].to(self.device)
        output = self.model(input_ids=valid_input_ids, 
                            attention_mask=valid_attention_mask, 
                            labels=valid_labels)
        valid_loss = output.loss
        valid_total_loss += valid_loss.item()
      return valid_total_loss/ len(self.valid_loader)
    
  def train(self):
    for epoch in range(self.n_epochs):
      train_total_loss = 0
      valid_mean_loss = None
      loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
      
      for batch_id, train_batch in loop:
        self.optimizer.zero_grad()
        self.global_step += 1
        train_input_ids = train_batch['input_ids'].to(self.device)
        train_attention_mask = train_batch['attention_mask'].to(self.device)
        train_labels = train_batch['label'].to(self.device)
        """ forward """
        output = self.model(input_ids=train_input_ids, 
                            attention_mask=train_attention_mask, 
                            labels =train_labels)
        train_loss = output.loss
        train_total_loss += train_loss.item()
        """ record training log """
        if self.tensorboard_writer != None:
          self.tensorboard_writer.add_scalar("train/loss", train_total_loss/ (batch_id+1), self.global_step)
        """ backward """
        train_loss.backward()
        """ update """
        self.optimizer.step()
        
        """ validation loss at end of epoch"""
        if self.valid_loader != None and batch_id == len(self.train_loader) - 1:
          valid_mean_loss = self.evaluate()
          if self.tensorboard_writer != None:
            self.tensorboard_writer.add_scalar("valid/loss", valid_mean_loss, self.global_step)
        """ print """
        train_mean_loss = train_total_loss / (batch_id+1)
        loop.set_description(f'Epoch [{epoch + 1}/{self.n_epochs}]')
        loop.set_postfix(train_loss=train_mean_loss, valid_loss=valid_mean_loss)
        
      """ end of epoch """
      # Save checkpoint
      if self.save_model_mode == 'all':
        self.save_model(epoch, train_mean_loss, valid_mean_loss)
      elif self.save_model_mode == 'best':
        if epoch == 0 or valid_mean_loss < self.best_loss:
          self.save_model(epoch, train_mean_loss, valid_mean_loss)
          
      # check early stop
      if self.early_stop:
        if self.loss_no_drop_epochs == self.early_stop_epochs - 1:
          break
        
        if valid_mean_loss > self.best_loss:
          self.loss_no_drop_epochs += 1
        else:
          self.loss_no_drop_epochs = 0    
          
      # reset best loss
      self.best_loss = min(self.best_loss, valid_mean_loss)
            

class TC_Predictor:
  def __init__(self, 
               model:AutoModelForSequenceClassification,
               tokenizer:AutoTokenizer, 
               dataset: Dataset,
               label_mode:str,
               label_map:Dict,
               batch_size:int,
               force_predict:bool=False,
               cutoff:Dict[str,float]=None,
               device:str=None):
    """
    This class inputs a fine-tuned model and a dataset. 
    outputs a list of TCs with entities (same as input), relations and probability
    {relation_id, relation_type, relation_prob, entity_1_id, entity_2_id, entity_1_text, entity_2_text}

    Parameters
    ----------
    model : AutoModelForSequenceClassification
      A model to make prediction.
    tokenizer : AutoTokenizer
      tokenizer.
    dataset : Dataset
      an (unlabeled) dataset for prediction.
    label_map : Dict
      DESCRIPTION.
    batch_size : int
      batch size for prediction. Does not affect prediction results.
    force_predict : bool
      For multi_label only, when none of the label has > 0.5 prob, force select
      the label with max prob.
    cutoff : Dict[str,float]
      For multi_label only, when a dict of cut-off are provided, use the cut-offs
      if None, use default 0.5 for all levels.
    device : str, optional
      CUDA device name. The default is cuda:0 if available, or cpu.
    """
    if device:
      self.device = device
    else: 
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      
    self.model = model
    self.model.to(self.device)
    self.model.eval()
    self.tokenizer = tokenizer
    self.label_mode = label_mode
    self.label_map = label_map
    self.batch_size = batch_size
    self.force_predict = force_predict
    self.cutoff =cutoff
    self.dataset = dataset
    self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
    

  def _predict_single_label(self) -> List[Text_Classification_Document]:
    tcs = {tc['doc_id']: Text_Classification_Document(doc_id=tc['doc_id'], 
                                                      text=tc['text']) 
           for tc in self.dataset.TCs}
    
    with torch.no_grad():
      loop = tqdm(self.dataloader, total=len(self.dataloader), leave=True)
      for batch in loop:  
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        p = self.model(input_ids=input_ids, attention_mask=attention_mask)
        batch_probs = p.logits.softmax(-1).cpu()
        doc_id_list = batch.doc_id
        for i, doc_id in enumerate(doc_id_list):
          tc = tcs[doc_id]
          probs = batch_probs[i]
          pred_prob = {k:p.item() for k, p in zip(self.label_map.keys(), probs)}
          label = max(pred_prob, key=lambda k: pred_prob[k])
          
          tc.label = label
          tc.predicted_prob = pred_prob
          tc.label_mode = 'single-label'
          
    return list(tcs.values())
    
  
  def _predict_multi_label(self) -> List[Text_Classification_Document]:
    tcs = {tc['doc_id']: Text_Classification_Document(doc_id=tc['doc_id'], 
                                                      text=tc['text']) 
           for tc in self.dataset.TCs}
    
    with torch.no_grad():
      loop = tqdm(self.dataloader, total=len(self.dataloader), leave=True)
      for batch in loop:  
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        p = self.model(input_ids=input_ids, attention_mask=attention_mask)
        batch_probs = p.logits.sigmoid().cpu()
        doc_id_list = batch.doc_id
        for i, doc_id in enumerate(doc_id_list):
          tc = tcs[doc_id]
          probs = batch_probs[i]
          pred_prob = {k:p.item() for k, p in zip(self.label_map.keys(), probs)}
          if self.cutoff is not None:
            label = [k for k, p in pred_prob.items() if p > self.cutoff[k]]
          else:
            label = [k for k, p in pred_prob.items() if p > 0.5]
          
          if len(label) == 0 and self.force_predict:
            label = [max(pred_prob.items(), key=lambda x:x[1])[0]]
            
          tc.label = label
          tc.predicted_prob = pred_prob
          tc.label_mode = 'multi-label'
          
    return list(tcs.values())
  
  
  def predict(self) -> List[Text_Classification_Document]:
    """
    This method outputs a list of TCs with label and predicted_prob
    """
    if self.label_mode == 'single-label':
      return self._predict_single_label()
    elif self.label_mode == 'multi-label':
      return self._predict_multi_label()
    

class TC_Evaluator:
  def __init__(self, pred_TCs:List[Text_Classification_Document],
               gold_TCs:List[Text_Classification_Document],
               label_mode:str):
    """
    This class inputs a list of predicted TCs and corresponding gold TCs
    outputs a 2-tuple of overall metrics and document-level evaluation.

    Parameters
    ----------
    pred_TCs : List[Text_Classification_Document]
      predicted TCs. Order doesn't matter.
    gold_TCs : List[Text_Classification_Document]
      gold TCs. Order doesn't matter. Could be more, but must cover all predicted TCs. 
    label_mode : str
      single-label or multi-label.
    """
    
    # Check all TCs have label and predicted probability
    self.has_prob = True
    for tc in pred_TCs:
      assert tc.has_label(), "All predicted TCs must have label."
      if not tc.has_predicted_prob():
        self.has_prob = False
        warnings.warn("Some or all TCs do not have predicted_prob." + \
                      "Probability-based evaluation metrics will not be calculated.", 
                      category=Warning)
        
    # Check if gold TCs have label
    for tc in gold_TCs:
      assert tc.has_label(), "All Gold TCs must have label."
      
    # Check if each predicted TC has a gold TC
    assert {tc['doc_id'] for tc in pred_TCs}.issubset({tc['doc_id'] for tc in gold_TCs}), \
        "Each predicted TC must have a matching gold TC."
    
    self.pred_TCs = pred_TCs
    self.gold_TCs = gold_TCs
    self.label_mode = label_mode
    self.gold_TC_dict = {tc['doc_id']:tc for tc in gold_TCs}
    if self.label_mode == 'single-label':
      self.label_types = [tc['label'] for tc in gold_TCs]
    elif self.label_mode == 'multi-label':
      self.label_types = []
      for tc in gold_TCs:
        self.label_types.extend(tc['label'])
        
  
  def F1(self, p:float, r:float):
    if p+r == 0:
      return float('nan')
    
    return 2*p*r/(p+r)
  
  
  def evaluate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    This function evaluates each label types available in gold
    Outputs a 2-tuple:
    overall metrics: a pd.DataFrame with {gold, pred, precision, recall, F1, (AUROC)}.
    each row is a label type.
    document-level evaluation: a pd.DataFrame with {doc_id, [label]_gold, [label]_pred}
    """
    eval_metrics = {l:{'label_type':l, 'gold':0, 'pred':0, 'match':0, 'precision':float('nan'), 
                       'recall':float('nan'), 'F1':float('nan')} for l in self.label_types}
    
    doc_df = pd.DataFrame({'doc_id':[tc['doc_id'] for tc in self.pred_TCs]})
    for l in self.label_types:
      pred_arr = []
      gold_arr = []
      prob_arr = []
      for pred_tc in self.pred_TCs:
        gold_tc = self.gold_TC_dict[pred_tc['doc_id']]
        if self.label_mode == 'single-label':
          pred_arr.append(pred_tc['label'] == l)
          gold_arr.append(gold_tc['label'] == l)
        elif self.label_mode == 'multi-label':
          pred_arr.append(l in pred_tc['label'])
          gold_arr.append(l in gold_tc['label'])
          
        if self.has_prob:
          prob_arr.append(pred_tc['predicted_prob'][l])
      
      # document-level 
      doc_df[f'{l}_gold'] = gold_arr
      doc_df[f'{l}_gold'] = doc_df[f'{l}_gold'].astype(int)
      doc_df[f'{l}_pred'] = pred_arr
      doc_df[f'{l}_pred'] = doc_df[f'{l}_pred'].astype(int)
      # overall metrics      
      eval_metrics[l]['gold'] = sum(gold_arr)
      eval_metrics[l]['pred'] = sum(pred_arr)
      eval_metrics[l]['precision'] = metrics.precision_score(gold_arr, pred_arr)
      eval_metrics[l]['recall'] = metrics.recall_score(gold_arr, pred_arr)
      eval_metrics[l]['F1'] = metrics.f1_score(gold_arr, pred_arr)
      if self.has_prob:
        eval_metrics[l]['AUROC'] = metrics.roc_auc_score(gold_arr, prob_arr)
  
    return pd.DataFrame([r for _, r in eval_metrics.items()]), doc_df
  
  