# Named Entity Recognizer
Earlier today, I was browsing through the list of Thesis Projects from past fourth-year EngScis, and one topic in particular stood out to me. Perhaps it was because I didn't know anything about it, but the words "Named Entity Recognition" remained stuck in my head for the rest of the day. As such, I decided to search up what it was and how to do it. Turns out, it's not actually too difficult. Subsequently, I used Spacy to train a custom Named Entity Recognizer to classify words from the light novel series Eighty Six. I also wondered what it would take to train an NER model from scratch, so I did just that using the CoNLL-2003 benchmark dataset. Unlike the fine-tuned model used for Eighty Six, this one is trained from scratch.

## Prerequisites

Remember to unzip the model folders!

```
datasets==1.16.1
python==3.8.0
spacy==3.2.0
```

## Dataset

<a href="https://huggingface.co/datasets/conll2003">CoNLL-2003</a>

## Results, with Pretrained Model

### Before Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/86_untrained_result.PNG">

### After Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/86_trained_result.PNG">

## Results, Training From Scratch with CoNLL-2003 Dataset

### Before Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/spacy_untrained_result.PNG">

### After Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/spacy_trained_result.PNG">

## Notes
* There is a weird bug with Spacy, where you can train, save, and load your model just fine.
* But then when you try to load your saved model without training first, it doesn't work.
* I'm not too sure why...

## Acknowledgements
* Varun Singh, for writing this <a href="https://keras.io/examples/nlp/ner_transformers/">Transformer tutorial</a>
* Shrivarsheni, for writing this <a href="https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/">article</a> on the basics of training Spacy NER models
* Asato Asato, for writing the amazing series that is Eighty Six
