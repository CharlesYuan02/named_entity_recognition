# Named Entity Recognizer
#### (named_entity_recognition.py)
Earlier today, I was browsing through the list of Thesis Projects from past fourth-year EngScis, and one topic in particular stood out to me. Perhaps it was because I didn't know anything about it, but the words "Named Entity Recognition" remained stuck in my head for the rest of the day. As such, I decided to search up what it was and how to do it. Turns out, it's not actually too difficult. Subsequently, I used Spacy to train a custom Named Entity Recognizer to classify words from the light novel series Eighty Six.

#### Update 1 (spacy_conll2003.py):
I wondered what it would take to train an NER model from scratch, so I did just that using the CoNLL-2003 benchmark dataset. Unlike the fine-tuned model used for Eighty Six, this one is trained from scratch. 

#### Update 2 (transformer_conll2003.py):
Of course, I can't conclude this project without testing out my own deep learning model. So I trained a transformer to compare with Spacy's model, and it actually did perform better! It ended up detecting all the entities in the text, along with identifying them with the correct labels.

## Prerequisites
```
datasets==1.16.1
python==3.8.0
spacy==3.2.0
```
## Dataset

The dataset I used was <a href="https://huggingface.co/datasets/conll2003">CoNLL-2003</a>, a named entity recognition dataset released as a part of CoNLL-2003 shared task: language-independent named entity recognition. The data was taken from the Reuters Corpus, which consists of Reuters news stories between August 1996 and August 1997.

## Results, with Pretrained Model and Custom Examples

### Before Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/results/86_untrained_result.PNG">

### After Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/results/86_trained_result.PNG">

## Results, Training From Scratch with CoNLL-2003 Dataset

#### Precision: 46.2% 
#### Recall: 51.7% 
#### F1 Score: 48.8% 

### Before Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/results/spacy_untrained_result.PNG">

### After Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/results/spacy_trained_result.PNG">

## Results, Transformer Model with CoNLL-2003 Dataset

#### Precision: 69.02% 
#### Recall: 65.58%
#### F1 Score: 67.26%

<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/results/transformer_trained_result.PNG">

<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/results/ground_truth.PNG">

## Notes
* There is a weird bug with Spacy, where you can train, save, and load your model just fine.
* But then when you try to load your saved model without training first, it doesn't work.
* I'm not too sure why...

## Acknowledgements
* Varun Singh, for writing this <a href="https://keras.io/examples/nlp/ner_transformers/">Transformer tutorial</a>
* Shrivarsheni, for writing this <a href="https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/">article</a> on the basics of training Spacy NER models
* Asato Asato, for writing the amazing series that is Eighty Six
