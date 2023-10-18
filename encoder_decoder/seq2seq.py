import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.legacy.data import Field, Example, Dataset, BucketIterator

spacy_en = spacy.load("en_core_web_sm")

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

context_field = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True, init_token='<sos>', eos_token='<eos>')
question_field = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True, init_token='<sos>', eos_token='<eos>')
answer_field = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True, init_token='<sos>', eos_token='<eos>')

fields = {'context': ('c', context_field), 'question': ('q', question_field), 'answer': ('a', answer_field)}

train_data, valid_data, test_data = Dataset.splits(
    path='./data/squad/',
    train='train.json',
    validation='valid.json',
    test='test.json',
    format='json',
    fields=fields
)

context_field.build_vocab(train_data, min_freq=2)
question_field.build_vocab(train_data, min_freq=2)
answer_field.build_vocab(train_data, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        return prediction, hidden, cell, a

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

input_dim = len(context_field.vocab)
output_dim = len(answer_field.vocab)
enc_emb_dim = 256
dec_emb_dim = 256
enc_hid_dim = 512
dec_hid_dim = 512
n_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

attention = Attention(enc_hid_dim, dec_hid_dim)
encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, n_layers, enc_dropout)
decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dec_dropout, attention)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(encoder, decoder, device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
