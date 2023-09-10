# NLP_TC_Pipelines
This is a general Natural Language Processing (NLP) system for text classification. It supports both multi-class (single-label) and multi-label classification tasks. The Text_Classification_Document(TC) class is the main data structure used through out the training, evaluation, and prediction. 

## Development pipeline overview
The  annotations are first converted to TC, then loaded by Dataset (PyTorch) to create training instances. 
![alt text](https://github.com/daviden1013/NLP_TC_Pipelines/blob/main/Development%20pipeline%20overview.png)

## Prediction pipeline overview
The raw text for information extraction is loaded and converted into TC. Then a fine-tuned text classification model makes prediction on the TCs and outputs TCs with labels and probabilities. 
![alt text](https://github.com/daviden1013/NLP_TC_Pipelines/blob/main/Prediction%20pipeline%20overview.png)
