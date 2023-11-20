import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
from tokenizers import BertWordPieceTokenizer

# slowTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# save_path = '../Data/bert_base_uncased/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# slowTokenizer.save_pretrained(save_path)

# loading the tokenizer from the saved file
tokenizer = BertWordPieceTokenizer('../Data/bert_base_uncased/vocab.txt', lowercase=True)
maxLength = 384
class SQUADExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    """
    def __init__(self, context, question, basic_answer, more_answers, startingIdx):
        self.context = context
        self.question = question
        self.basic_answer = basic_answer
        self.more_answers = more_answers
        self.startingIdx = startingIdx
        self.endingIdx = None
        self.attention_mask = None
        self.input_ids = None
        self.tokenTypeIds = None
        self.offSets = None
        self.validExample = True
        self.startIdxtoken = startingIdx
        self.endIndextoken = None

    def preProcessing(self):
        newContext = str(self.context).lower().split()
        self.context = ' '.join(newContext)
        newQuestion = str(self.question).lower().split()
        self.question = ' '.join(newQuestion)
        contextTokens = tokenizer.encode(self.context)
        if self.basic_answer is not None :
            # if we have answer
            self.basic_answer = ' '.join(str(self.basic_answer).lower().split())
            self.endingIdx = self.startingIdx + len(self.basic_answer)
            if self.endingIdx >= len(self.context):
                self.validExample = False
                return

            # iterate from start to end to find the characters of context
            isPartOfAnswer = [0] * len(self.context)
            for idx in range(self.startingIdx, self.endingIdx):
                isPartOfAnswer[idx] = 1

            answerIdToken = []
            for idx, (start, end) in enumerate(contextTokens.offsets):
                if sum(isPartOfAnswer[start:end]) > 0:
                    answerIdToken.append(idx)
            # data to predict the start and end index of the answer
            if len(answerIdToken) == 0:
                self.validExample = False
                return
            self.startIdxtoken = answerIdToken[0]
            self.endIndextoken = answerIdToken[-1]

        self.offSets = contextTokens.offsets
        questionTokenizer = tokenizer.encode(self.question)
        self.input_ids = contextTokens.ids + questionTokenizer.ids[1:]
        self.attention_mask = [1] * len(self.input_ids)
        self.tokenTypeIds = [0] * len(contextTokens.ids) + [1] * len(questionTokenizer.ids[1:])

        # padding fixing
        paddingLength = maxLength - len(self.input_ids)
        if paddingLength > 0:
            self.input_ids = self.input_ids + ([0] * paddingLength)
            self.attention_mask = self.attention_mask + ([0] * paddingLength)
            self.tokenTypeIds = self.tokenTypeIds + ([0] * paddingLength)
        elif paddingLength < 0:
            self.validExample = False
            return