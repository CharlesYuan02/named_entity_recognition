# Named Entity Recognizer
Earlier today, I was browsing through the list of Thesis Projects from past fourth-year EngScis, and one topic in particular stood out to me. Perhaps it was because I didn't know anything about it, but the words "Named Entity Recognition" remained stuck in my head for the rest of the day. As such, I decided to search up what it was and how to do it. Turns out, it's not actually too difficult. Subsequently, I used Spacy to train a custom Named Entity Recognizer to classify words from the light novel series Eighty Six.

## Prerequisites

Remember to unzip the model folder!

```
python==3.8.0
spacy==3.2.0
```

## Results

### Before Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/untrained_result.PNG">

### After Training
<img src="https://github.com/Chubbyman2/named_entity_recognition/blob/main/trained_result.PNG">

## Acknowledgements
* Shrivarsheni, for writing this <a href="https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/">article</a> on the basics of training Spacy NER models
* Asato Asato, for writing the amazing series that is Eighty Six
