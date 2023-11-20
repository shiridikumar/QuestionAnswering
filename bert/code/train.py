import string
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from preProcessing import SQUADExample
from featureExtraction import createSquadExamples, createInputsTargets
import numpy as np
import json 
from transformers import BertForQuestionAnswering



DOC_STRIDE = 64
MAX_SEQ_LENGTH = 128
MAX_QUERY_LENGTH = 32
BATCH_SIZE = 16

train_data_path = '../Data/train-v2.0.json'
test_data_path = '../Data/dev-v2.0.json'

def readDataFile(train_path, valid_path):
    with open(train_path, 'r') as f:
        raw_train_data = json.load(f)
    with open(valid_path, 'r') as f:
        raw_valid_data = json.load(f)

    return raw_train_data, raw_valid_data


raw_train_data, raw_valid_data = readDataFile(train_data_path, test_data_path)
train_data = createSquadExamples(raw_train_data)
valid_data = createSquadExamples(raw_valid_data)

X_train, y_train = createInputsTargets(train_data)
X_val, y_val = createInputsTargets(valid_data)


trainData = TensorDataset(torch.tensor(X_train[0], dtype=torch.int64),
                           torch.tensor(X_train[1], dtype=torch.float),
                           torch.tensor(X_train[2], dtype=torch.int64),
                           torch.tensor(y_train[0], dtype=torch.int64),
                           torch.tensor(y_train[1], dtype=torch.int64))

train_sampler = RandomSampler(trainData)
train_dataloader = DataLoader(
    trainData, sampler=train_sampler, batch_size=BATCH_SIZE)

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

validData = TensorDataset(torch.tensor(X_val[0], dtype=torch.int64),
                            torch.tensor(X_val[1], dtype=torch.float),
                            torch.tensor(X_val[2], dtype=torch.int64),
                            torch.tensor(y_val[0], dtype=torch.int64),
                            torch.tensor(y_val[1], dtype=torch.int64))
valid_sampler = SequentialSampler(validData)
valid_dataloader = DataLoader(
    validData, sampler=valid_sampler, batch_size=BATCH_SIZE)


################# training code #################
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = torch.optim.Adam(
    lr=1e-5, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)


def normalizeText(text):
    if text is None or len(text)==0:
      return ""
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE) # remove a, an, the
    text = re.sub(regex, ' ', text)
    text = " ".join(text.split())
    return text


from tqdm import tqdm, trange
def train(model, train_dataloader, val_data, validation_dataloader, optimizer, epochs=2, max_grad_norm=1.0):
    model.train()
    for _ in trange(epochs, desc='Epoch'):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'start_positions': batch[3], 'end_positions': batch[4]}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            torch.nn.utils.clip_grad_norm_( parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            model.zero_grad()
            if(step % 100 == 0):
              print("Batch loss : {}".format(tr_loss/nb_tr_steps))
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        validate(model, val_data, validation_dataloader)

    print("Training complete")

def validate(model, val_data, validation_dataloader):
    model.eval()
    currentQuery = 0
    correctAns = 0
    validExamples = [x for x in val_data if x.validExample]
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        with torch.no_grad():
            start_scores, end_scores = model(input_ids, token_type_ids=segment_ids,
                                             attention_mask=input_mask,return_dict=False)

            pred_start, pred_end = start_scores.detach().cpu().numpy(), end_scores.detach().cpu().numpy()

        for idx, (start,end) in enumerate(zip(pred_start,pred_end)):
            squadEg = validExamples[currentQuery]
            currentQuery +=1
            offsets = squadEg.offSets
            startIdx = np.argmax(start)
            endIdx = np.argmax(end)
            if startIdx >= len(offsets):
                continue
            predCharStart = offsets[startIdx][0]
            if endIdx < len(offsets):
                predCharEnd = offsets[endIdx][1]
                predAnswer = squadEg.context[predCharStart:predCharEnd]
            else:
                predAnswer = squadEg.context[predCharStart:]
            if(predAnswer==None):
              continue
            normalizedPredAnswer = normalizeText(predAnswer)
            normalizedTrueAnswer = [normalizeText(x)
                                    for x in squadEg.more_answers]
            normalizedTrueAnswer.append(normalizeText(squadEg.basic_answer))
            if normalizedPredAnswer in normalizedTrueAnswer:
                correctAns += 1
            if(currentQuery + idx) % 50 == 0:
              print("Validated {}/{} examples".format(currentQuery+idx+1, len(validExamples)))
    acc = correctAns / len(validExamples)
    print("Validation Accuracy: {}".format(acc))
    return acc


train(model, train_dataloader, valid_data, valid_dataloader, optimizer, epochs=1, max_grad_norm=1.0)
model.save_pretrained('/content/drive/My Drive/BERT/model')
