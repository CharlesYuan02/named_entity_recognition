import numpy as np
import os
import tensorflow as tf
from collections import Counter
from conlleval import evaluate
from datasets import load_dataset
from spacy import displacy
from spacy_conll2003 import export_to_file, create_lookup_table
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    '''
    Basic transformer architecture, see original paper for more info.
    '''
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings


class NERModel(tf.keras.Model):
    '''
    This is the transformer model itself.
    '''
    def __init__(
        self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32
    ):
        super(NERModel, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x


def map_record_to_training_data(record):
    '''
    Takes in a line of data and returns the tokens and tags.
    '''
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1:length+1]
    tags = record[length+1:]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1
    return tokens, tags


def lowercase_and_convert_to_ids(tokens):
    tokens = tf.strings.lower(tokens)
    return lookup_layer(tokens)


def tokenize_and_convert_to_ids(text):
        tokens = text.split()
        return lowercase_and_convert_to_ids(tokens)


class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
    '''
    Sparse Categorical Crossentropy ignores loss from padded tokens.
    '''
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def calculate_metrics(dataset, model, lookup_table):
    '''
    Returns final precision, recall, F1 Score.
    '''
    all_true_tag_ids, all_predicted_tag_ids = [], []

    for x, y in dataset:
        output = model.predict(x)
        predictions = np.argmax(output, axis=-1)
        predictions = np.reshape(predictions, [-1])

        true_tag_ids = np.reshape(y, [-1])

        mask = (true_tag_ids > 0) & (predictions > 0)
        true_tag_ids = true_tag_ids[mask]
        predicted_tag_ids = predictions[mask]

        all_true_tag_ids.append(true_tag_ids)
        all_predicted_tag_ids.append(predicted_tag_ids)

    all_true_tag_ids = np.concatenate(all_true_tag_ids)
    all_predicted_tag_ids = np.concatenate(all_predicted_tag_ids)

    predicted_tags = [lookup_table[tag] for tag in all_predicted_tag_ids]
    real_tags = [lookup_table[tag] for tag in all_true_tag_ids]

    evaluate(real_tags, predicted_tags)


def return_displacy_render(text, label_array, title=None):
    '''
    This will return the sample text in the form that can be displayed using displacy.
    text is a string of text.
    label_array is what's predicted by the transformer (e.g. ['O', 'B-ORG', 'O']).
    title is any string that you want.
    '''
    ent_array = []
    word_array = text.split()

    for i in range(len(label_array)):
        if label_array[i] != "O" and label_array[i] != "[PAD]":
            word = word_array[i]
            start = text.find(word)
            end = start + len(word)
            label = label_array[i]
            ent_array.append({"start": start, "end": end, "label": label})
    
    ex = [{"text": text,
           "ents": ent_array,
           "title": title}]
    
    return ex


if __name__ == "__main__":
    # We're gonna use the CoNLL-2003 dataset again
    conll_data = load_dataset("conll2003")
    if not os.path.exists("data"):
        os.mkdir("data")
        export_to_file("./data/conll_train.txt", conll_data["train"])
        export_to_file("./data/conll_val.txt", conll_data["validation"])

    lookup_table = create_lookup_table()

    all_tokens = sum(conll_data["train"]["tokens"], [])
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))
    print(all_tokens_array)

    counter = Counter(all_tokens_array)
    print(len(counter))

    num_tags = len(lookup_table)
    vocab_size = 20000

    # The minus two is because StringLookup class uses 2 additional tokens
    # One for an unknown token and the other one is a masking token
    vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]

    # Use StringLookup class to convert tokens to token IDs
    lookup_layer = tf.keras.layers.StringLookup(vocabulary=vocabulary)

    # Convert datasets to Dataset objects for training
    train_data = tf.data.TextLineDataset("./data/conll_train.txt")
    val_data = tf.data.TextLineDataset("./data/conll_val.txt")
    print(list(train_data.take(1).as_numpy_iterator()))

    batch_size = 32
    train_dataset = (
        train_data.map(map_record_to_training_data)
        .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
        .padded_batch(batch_size) # The padded batch is to compensate for different text lengths
    )
    val_dataset = (
        val_data.map(map_record_to_training_data)
        .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
        .padded_batch(batch_size)
    )

    # Create model, set loss function
    model = NERModel(num_tags, vocab_size, embed_dim=32, num_heads=4, ff_dim=64)
    loss = CustomNonPaddingTokenLoss()

    # Train Model
    model.compile(optimizer="adam", loss=loss)
    model.fit(train_dataset, epochs=30)

    # Sample inference using the trained model
    example = "West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship."
    sample_input = tokenize_and_convert_to_ids(example)
    sample_input = tf.reshape(sample_input, shape=[1, -1])
    print(sample_input)

    output = model.predict(sample_input)
    prediction = np.argmax(output, axis=-1)[0]
    prediction = [lookup_table[i] for i in prediction]

    calculate_metrics(val_dataset, model, lookup_table)

    # We can actually use displacy to manually render data
    # Just pass the sample text and prediction into the function I wrote
    ex = return_displacy_render(example, prediction, "Transformer Output")

    colors = {"B-PER": "#FF6D6D", # Red
              "I-PER": "#FF6D6D",
              "B-ORG": "#FFB454", # Orange
              "I-ORG": "#FFB454",
              "B-LOC": "#81FF4A", # Green 
              "I-LOC": "#81FF4A",
              "B-MISC": "#4ABDFF", # Blue
              "I-MISC": "#4ABDFF"}
    options = {"ents": ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"], "colors": colors}
    
    # To see it, open up "http://localhost:5000/"
    displacy.serve(ex, style="ent", options=options, manual=True)


