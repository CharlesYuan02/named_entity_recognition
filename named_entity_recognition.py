import random
import spacy
from pathlib import Path
from spacy import displacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example

nlp=spacy.load('en_core_web_sm')
# print(nlp.pipe_names) # Make sure you have 'ner' available

def create_train_data(text, word, entity_type):
    '''
    Generates a training datum from a sample piece of text in the following form:
    ("text", {"entities: [(start, end, "entity_type")]})
    e.g.
    ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]})
    start and end are found based on the word given
    '''
    start = text.find(word)
    end = start + len(word)

    ret = (text, {"entities": [(start, end, entity_type)]})
    return ret


def train_ner(train_data, ner):
    for _, annotations in train_data:
        for entity in annotations.get("entities"):
            ner.add_label(entity[2])
    
    # Disable other pipeline components which shouldn't be affected
    trained_components = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in trained_components]

    with nlp.disable_pipes(*unaffected_pipes):
        for iter in range(30):
            random.shuffle(train_data) # Good to shuffle examples each iteration
            losses = {}

            # Create batches using minibatch
            batches = minibatch(train_data, size=2)
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                nlp.update([example], losses=losses, drop=0.5)
                print(f"Losses: {losses}")


if __name__ == "__main__":
    # Here is the sample text from 86 Vol 3
    example = ("'I can't give her my name yet.' \nNot when he was only seeking a place to die "
              "and hadn't progressed at all since the Eighty-Sixth Sector."
              "\n'If she says she's caught up, I can't let this be what she sees at the end of that road. "
              "What she sees when she catches up to us shouldn't be...' "
              "\nHim kneeling down on this crumbled earth. \n'...shouldn't be this battlefield.' "
              "\nFrederica sighed with astonishment. \n'How do I put this? You truly are a boy after all.'")
    
    doc = nlp(example)
    for entity in doc.ents:
        print(entity.text, entity.label_)
    # As you can see, the untrained model only recognized "Frederica" as an entity.
    # We want it to recognize more.

    ner = nlp.get_pipe("ner")

    # But in order to train our custom model, we have to have custom data
    # Luckily, I've written an easy function to easily generate data in the correct format
    data1 = create_train_data("The Republic's Eighty-Sixth Sector was located in the northern part of the continent and would often get chilly after sunset.", "Eighty-Sixth Sector", "GPE")
    print(data1) # Take a look

    '''
    List of entity tags for reference:
    GEO = Geographical Entity
    ORG = Organization
    PER = Person
    GPE = Geopolitical Entity
    TIM = Time indicator
    ART = Artifact
    EVE = Event
    NAT = Natural Phenomenon
    '''
    
    # Now I'll write the rest
    # Yes, these are all real entries from the light novels
    data2 = create_train_data("The majority of the Eighty-Six we took under our protection were what they cal Name Bearers - veterans who lived through years in the Eighty-Sixth Sector's battlefield despite the 0.1 survival rate.", "Eighty-Sixth Sector", "GPE")
    data3 = create_train_data("Being defenseless on the battlefield like this was an act of recklessness.", "battlefield", "GEO")
    data4 = create_train_data("They survived this long on the battlefield, and they know if they intend to keep fighting, our survival is necessary.", "battlefield", "GEO")
    data5 = create_train_data("Not a single trace of life remained on this battlefield except for him.", "battlefield", "GEO")
    data6 = create_train_data("There were no casualties on that battlefield.", "battlefield", "GEO")
    data7 = create_train_data("Frederica unsteadily rose to her feet.", "Frederica", "PER")
    data8 = create_train_data("Frederica's body stiffened upon hearing that voice.", "Frederica", "PER")

    # Append them all to the training array
    train = [data1, data2, data3, data4, data5, data6, data7, data8]

    # Train the named entity recognizer
    train_ner(train, ner)

    # Now let's test it
    new_doc = nlp(example)
    for entity in new_doc.ents:
        print(entity.text, entity.label_)

    # The results look much better! 
    # As you can see, selecting specific excerpts for the words I wanted identified worked.
    # This is model training 101 (a.k.a. bias manipulation)

    # Now let's save this model
    output_dir=Path("C:\\Users\\charl\\Desktop\\named_entity_recognition\\model\\")
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.meta["name"] = "eighty_six_ner" # Feel free to name this anything you want
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}.")

    # Now let's load this model and see if it still works
    nlp_test = spacy.load(output_dir)
    move_names = list(ner.move_names)
    assert nlp_test.get_pipe("ner").move_names == move_names
    test_doc = nlp_test("'I can't give her my name yet.' \nNot when he was only seeking a place to die "
              "and hadn't progressed at all since the Eighty-Sixth Sector."
              "\n'If she says she's caught up, I can't let this be what she sees at the end of that road. "
              "What she sees when she catches up to us shouldn't be...' "
              "\nHim kneeling down on this crumbled earth. \n'...shouldn't be this battlefield.' "
              "\nFrederica sighed with astonishment. \n'How do I put this? You truly are a boy after all.'")
    for entity in test_doc.ents:
        print(entity.text, entity.label_)


    # You can visualize the final results in a nicer format
    # To see it, open up "http://localhost:5000/"
    displacy.serve(test_doc, style='ent')






