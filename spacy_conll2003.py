import os
import random
import spacy
from datasets import load_dataset
from pathlib import Path
from spacy import displacy
from spacy.training.example import Example 
from spacy.util import minibatch, compounding


def export_to_file(export_file_path, data):
    '''
    Creates a properly formatted training/validation txt file.
    We can then use these txt files to create our training data.
    '''
    with open(export_file_path, "w") as f:
        for record in data:
            ner_tags = record["ner_tags"]
            tokens = record["tokens"]
            f.write(
                str(len(tokens))
                + "\t"
                + "\t".join(tokens)
                + "\t"
                + "\t".join(map(str, ner_tags))
                + "\n"
            )


def create_lookup_table():
    '''
    The data is labelled with numbers, but we want actual text labels.
    The labels this time around will be slightly different for this dataset.
    B = Beginning
    I = Inside
    O = Non-named entities
    PERSON, ORGANIZATION, LOCATION, MISCELLANEOUS
    '''
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))


def create_train_data(text, tokens, tags, lookup_table):
    '''
    Generates a training datum from a sample piece of text in the following form:
    ("text", {"entities: [(start, end, "entity_type")]})
    This is just a slightly modified version of the one from named_entity_recognition.py,
    which now allows for multiple entities per line of text.
    '''
    entity_list = []
    no_repeats = [] # Just a list to store positions already seen to ensure no repeat labels

    for i in range(len(tokens)):
        start = text.find(tokens[i])
        end = start + len(tokens[i])
        entity_type = lookup_table[tags[i]]
        
        overlap = False
        # We don't want NER to recognize "O"
        if entity_type != "O" and (start, end) not in no_repeats:
            for j in range(len(no_repeats)):
                # This is to ensure overlaps between entities
                # i.e. '(26, 33, 'I-ORG')' and '(28, 30, 'I-LOC')' are overlaps
                if not ((start >= int(no_repeats[j][0]) and start <= int(no_repeats[j][1])) or (end >= int(no_repeats[j][0]) and end <= int(no_repeats[j][1]))):
                    pass
                else:
                    overlap = True
                    break
            
            if not overlap:
                entity_list.append((start, end, entity_type))
                no_repeats.append((start, end))
            else:
                overlap = False

    ret = (text, {"entities": entity_list})
    return ret


def train_ner(train_data, ner):
    for _, annotations in train_data:
        for entity in annotations.get("entities"):
            ner.add_label(entity[2])
    
    # Disable other pipeline components which shouldn't be affected
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*unaffected_pipes):
        for iter in range(2): # Since it's a lot of data (14k), train for only 2 iterations
            random.shuffle(train_data) # Good to shuffle examples each iteration
            losses = {}

            # Create batches using minibatch
            batches = minibatch(train_data, size=16)
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                nlp.update([example], losses=losses, drop=0.5)
                print(f"Losses: {losses}")


def evaluate(nlp, val_data, ents):
    scorer = Scorer()
    for input_, annotations in val_data:
        doc_gold_text = nlp.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=ents)
        pred_value = nlp(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

if __name__ == "__main__":
    # Since we're training from scratch, make sure this is not a pretrained model!
    nlp = spacy.blank('en')
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner")
    nlp.initialize()

    conll_data = load_dataset("conll2003")

    # Create training/validation data
    if not os.path.exists("data"):
        os.mkdir("data")
        export_to_file("./data/conll_train.txt", conll_data["train"])
        export_to_file("./data/conll_val.txt", conll_data["validation"])

    lookup_table = create_lookup_table()
    print(lookup_table)

    # Let's use one line of the validation examples as a testing case
    file_dir = "data/conll_val.txt"
    with open(file_dir, "r") as test_file:
        lines = test_file.readlines()
        line0 = lines[2]
    
    record = line0.split(sep="\t")
    length = int(record[0]) # record[0] is always an int giving the number of words
    example = " ".join(record[1:length])

    print(example)
    doc = nlp(example)
    print("\nBefore Training: ")
    for entity in doc.ents:
        print(entity.text, entity.label_)

    # As you can see, the untrained model obviously doesn't detect anything.
    # So now let's actually train it, shall we?    

    train_dir = "data/conll_train.txt"
    train_data = []
    
    with open(train_dir, "r") as train_file:
        lines = train_file.readlines()
        for i in range(len(lines)):
            # We want to create a training example for each line
            # To do that, we need to pass in the text, tokens, tags, and lookup_table to the function I wrote
            record = lines[i]
            record = record.split(sep="\t")
            length = int(record[0])
            tokens = record[1:length+1]
            tags = record[length+1: ]
            tags = [int(tag)+1 for tag in tags]
            text = " ".join(record[1:length+1])
            train_data.append(create_train_data(text, tokens, tags, lookup_table))

    print(f"Length of training data: {len(train_data)}")
    print("Sample datum: ")
    print(train_data[0])

    # As you can see, we have a total of 14041 examples for training
    # So now let's train it
    train_ner(train_data, ner)

    # Now let's save this model
    output_dir = Path("C:\\Users\\charl\\Desktop\\named_entity_recognition\\spacy_model\\")
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.meta["name"] = "conll2003_spacy" # Feel free to name this anything you want
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}.")

    # Now let's load this model and see if it still works
    nlp_test = spacy.load(output_dir)
    # print(nlp_test.pipe_names)
    ner = nlp_test.get_pipe("ner")
    test_doc = nlp_test(example)
    print("\nAfter Training: ")
    for entity in test_doc.ents:
        print(entity.text, entity.label_)

    # As you can see, the model is now able to perform named entity recognition!
    # Note that with enough examples, it can even do this on custom labels like the ones we added.

    # Last thing, let's evaluate its accuracy using the validation data!
    # First let's create the validation set
    val_dir = "data/conll_val.txt"
    val_data = []

    with open(val_dir, "r") as val_file:
        lines = val_file.readlines()
        for i in range(len(lines)):
            record = lines[i]
            record = record.split(sep="\t")
            length = int(record[0])
            tokens = record[1:length+1]
            tags = record[length+1: ]
            tags = [int(tag)+1 for tag in tags]
            text = " ".join(record[1:length+1])
            val_data.append(create_train_data(text, tokens, tags, lookup_table))
    
    # But the evaluate() method needs the data in the form of Example objects
    val_examples = []
    for text, annotations in val_data:
        val_doc = nlp_test.make_doc(text)
        example = Example.from_dict(val_doc, annotations)
        val_examples.append(example)
    
    # Now let's obtain the training scores
    scores = nlp.evaluate(val_examples)
    print(f"Precision: {scores['ents_p']} \nRecall: {scores['ents_r']} \nF1 Score: {scores['ents_f']}")
    
    # You can visualize the final results in a nicer format
    # Let's add some custom colours while we're at it
    # To see it, open up "http://localhost:5000/"
    colors = {"B-PER": "#FF6D6D", # Red
              "I-PER": "#FF6D6D",
              "B-ORG": "#FFB454", # Orange
              "I-ORG": "#FFB454",
              "B-LOC": "#81FF4A", # Green 
              "I-LOC": "#81FF4A",
              "B-MISC": "#4ABDFF", # Blue
              "I-MISC": "#4ABDFF"}
    options = {"ents": ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"], "colors": colors}

    displacy.serve(test_doc, style='ent', options=options)
