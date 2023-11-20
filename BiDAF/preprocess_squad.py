import spacy
import torch
import pandas as pd
import pickle
from preprocess import load_json, parse_data, gather_text_for_vocab, build_word_vocab, build_char_vocab, \
    get_error_indices, index_answer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load and preprocess training data
train_data = load_json('./data/squad_train.json')
train_list = parse_data(train_data)
train_df = pd.DataFrame(train_list)

# Load and preprocess validation data
valid_data = load_json('./data/squad_dev.json')
valid_list = parse_data(valid_data)
valid_df = pd.DataFrame(valid_list)

print('--------------------------')
print('Train list len: ', len(train_list))
print('Valid list len: ', len(valid_list))

def preprocess_df(df):
    def to_lower(text):
        return text.lower()

    df.context = df.context.apply(to_lower)
    df.question = df.question.apply(to_lower)
    df.answer = df.answer.apply(to_lower)

preprocess_df(train_df)
preprocess_df(valid_df)

# Build vocabulary and save to files
vocab_text = gather_text_for_vocab([train_df, valid_df])
word2idx, idx2word, word_vocab = build_word_vocab(vocab_text)
char2idx, char_vocab = build_char_vocab(vocab_text)

train_err = get_error_indices(train_df, idx2word)
valid_err = get_error_indices(valid_df, idx2word)

train_df.drop(train_err, inplace=True)
valid_df.drop(valid_err, inplace=True)

train_label_idx = train_df.apply(index_answer, axis=1, idx2word=idx2word)
valid_label_idx = valid_df.apply(index_answer, axis=1, idx2word=idx2word)

train_df['label_idx'] = train_label_idx
valid_df['label_idx'] = valid_label_idx

# Save preprocessed data to pickle files
train_df.to_pickle('bidaftrain.pkl')
valid_df.to_pickle('bidafvalid.pkl')

with open('bidafw2id.pickle', 'wb') as handle:
    pickle.dump(word2idx, handle)

with open('bidafc2id.pickle', 'wb') as handle:
    pickle.dump(char2idx, handle)

# Load preprocessed data and vocabulary
train_df = pd.read_pickle('bidaftrain.pkl')
valid_df = pd.read_pickle('bidafvalid.pkl')

with open('bidafw2id.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)

with open('bidafc2id.pickle', 'rb') as handle:
    char2idx = pickle.load(handle)

idx2word = {v: k for k, v in word2idx.items()}
