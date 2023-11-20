import torch
import torch.nn.functional as F
import spacy

nlp = spacy.load("en_core_web_sm")

class SquadDataset:
    def __init__(self, data, batch_size, word2idx, char2idx):
        self.batch_size = batch_size
        self.data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.word2idx = word2idx
        self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def make_char_vector(self, max_sent_len, max_word_len, sentence):
        char_vec = torch.ones(max_sent_len, max_word_len).type(torch.LongTensor)
        for i, word in enumerate(nlp(sentence, disable=['parser', 'tagger', 'ner'])):
            for j, ch in enumerate(word.text):
                char_vec[i][j] = self.char2idx.get(ch, 0)
        return char_vec

    def get_span(self, text):
        text = nlp(text, disable=['parser', 'tagger', 'ner'])
        span = [(w.idx, w.idx + len(w.text)) for w in text]
        return span

    def prepare_sequence(self, seq, to_idx, max_len):
        seq = [to_idx.get(word, 0) for word in seq]
        seq = seq[:max_len] + [0] * max(0, max_len - len(seq))
        return torch.LongTensor(seq)

    def prepare_char_sequence(self, seq, max_len, max_word_len):
        char_seq = torch.ones(max_len, max_word_len).type(torch.LongTensor)
        for i, word in enumerate(nlp(seq, disable=['parser', 'tagger', 'ner'])):
            for j, ch in enumerate(word.text):
                char_seq[i][j] = self.char2idx.get(ch, 0)
        return char_seq

    def process_batch(self, batch):
        spans = [self.get_span(ctx) for ctx in batch.context]
        ctx_text = list(batch.context)
        answer_text = list(batch.answer)

        max_context_len = max(len(ctx) for ctx in batch.context_ids)
        padded_context = torch.stack([self.prepare_sequence(ctx, self.word2idx, max_context_len) for ctx in batch.context_ids])

        max_word_ctx = max(len(word.text) for context in batch.context for word in nlp(context, disable=['parser', 'tagger', 'ner']))
        char_ctx = torch.stack([self.prepare_char_sequence(context, max_context_len, max_word_ctx) for context in batch.context])

        max_question_len = max(len(ques) for ques in batch.question_ids)
        padded_question = torch.stack([self.prepare_sequence(ques, self.word2idx, max_question_len) for ques in batch.question_ids])

        max_word_ques = max(len(word.text) for question in batch.question for word in nlp(question, disable=['parser', 'tagger', 'ner']))
        char_ques = torch.stack([self.prepare_char_sequence(question, max_question_len, max_word_ques) for question in batch.question])

        ids = list(batch.id)
        label = torch.LongTensor(list(batch.label_idx))

        return (padded_context, padded_question, char_ctx, char_ques, label, ctx_text, answer_text, ids)

    def __iter__(self):
        for batch in self.data:
            yield self.process_batch(batch)
