import torch
import numpy as np
import pandas as pd
import pickle
import re, os, string, typing, gc, json
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Length of data: ", len(data['data']))
    print("Data Keys: ", data['data'][0].keys())
    print("Title: ", data['data'][0]['title'])

    return data


def parse_data(data: dict) -> list:
    data = data['data']
    qa_list = []

    for paragraphs in data:
        for para in paragraphs['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                id = qa['id']
                question = qa['question']
                for ans in qa['answers']:
                    answer = ans['text']
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(answer)

                    qa_dict = {
                        'id': id,
                        'context': context,
                        'question': question,
                        'label': [ans_start, ans_end],
                        'answer': answer
                    }
                    qa_list.append(qa_dict)

    return qa_list


def filter_large_examples(df):
    ctx_lens, query_lens, ans_lens = [], [], []
    for index, row in df.iterrows():
        ctx_tokens = [w.text for w in nlp(row.context, disable=['parser', 'ner', 'tagger'])]
        if len(ctx_tokens) > 400:
            ctx_lens.append(row.name)

        query_tokens = [w.text for w in nlp(row.question, disable=['parser', 'tagger', 'ner'])]
        if len(query_tokens) > 50:
            query_lens.append(row.name)

        ans_tokens = [w.text for w in nlp(row.answer, disable=['parser', 'tagger', 'ner'])]
        if len(ans_tokens) > 30:
            ans_lens.append(row.name)

        assert row.name == index

    return set(ans_lens + ctx_lens + query_lens)


def gather_text_for_vocab(dfs: list):
    text = []
    total = 0
    for df in dfs:
        unique_contexts = list(df.context.unique())
        unique_questions = list(df.question.unique())
        total += df.context.nunique() + df.question.nunique()
        text.extend(unique_contexts + unique_questions)

    assert len(text) == total
    return text


def build_word_vocab(vocab_text):
    words = [word.text for sent in vocab_text for word in nlp(sent, disable=['parser', 'tagger', 'ner'])]
    word_counter = Counter(words)
    word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    word_vocab.insert(0, '<unk>')
    word_vocab.insert(1, '<pad>')
    word2idx = {word: idx for idx, word in enumerate(word_vocab)}
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word, word_vocab


def build_char_vocab(vocab_text):
    chars = [ch for sent in vocab_text for ch in sent]
    char_counter = Counter(chars)
    char_vocab = sorted(char_counter, key=char_counter.get, reverse=True)
    high_freq_char = [char for char, count in char_counter.items() if count >= 20]
    char_vocab = list(set(char_vocab).intersection(set(high_freq_char)))
    char_vocab.insert(0, '<unk>')
    char_vocab.insert(1, '<pad>')
    char2idx = {char: idx for idx, char in enumerate(char_vocab)}
    return char2idx, char_vocab


def context_to_ids(text, word2idx):
    context_tokens = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
    context_ids = [word2idx.get(word, 0) for word in context_tokens]
    return context_ids


def question_to_ids(text, word2idx):
    question_tokens = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
    question_ids = [word2idx.get(word, 0) for word in question_tokens]
    return question_ids


def test_indices(df, idx2word):
    start_value_error, end_value_error, assert_error = [], [], []
    for index, row in df.iterrows():
        answer_tokens = [w.text for w in nlp(row['answer'], disable=['parser', 'tagger', 'ner'])]
        start_token, end_token = answer_tokens[0], answer_tokens[-1]
        context_span = [(word.idx, word.idx + len(word.text))
                        for word in nlp(row['context'], disable=['parser', 'tagger', 'ner'])]
        starts, ends = zip(*context_span)
        answer_start, answer_end = row['label']
        try:
            start_idx = starts.index(answer_start)
        except ValueError:
            start_value_error.append(index)
        try:
            end_idx = ends.index(answer_end)
        except ValueError:
            end_value_error.append(index)
        try:
            assert idx2word[row['context_ids'][start_idx]] == answer_tokens[0]
            assert idx2word[row['context_ids'][end_idx]] == answer_tokens[-1]
        except AssertionError:
            assert_error.append(index)
    return start
