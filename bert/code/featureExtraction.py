from preProcessing import SQUADExample
import numpy as np

def createSquadExamples(raw_data):
    squadExamples = []
    for item in raw_data['data']:
        for para in item['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                question = qa['question']
                basic_answer = None
                more_answers = []
                startingIdx = None
                if qa['is_impossible']:
                    basic_answer = None
                else:
                    basic_answer = qa['answers'][0]['text']
                    startingIdx = qa['answers'][0]['answer_start']
                squadEg = SQUADExample(context, question, basic_answer, more_answers, startingIdx)
                squadEg.preProcessing()
                squadExamples.append(squadEg)
    return squadExamples

def createInputsTargets(squad_example):
    datasetDict = {}
    for item in squad_example:
        if item.validExample:
            for key in ['input_ids', 'attention_mask', 'tokenTypeIds', 'startIdxtoken', 'endIndextoken']:
                if key not in datasetDict:
                    datasetDict[key] = []
                datasetDict[key].append(item.__dict__[key])

    for key in datasetDict:
        datasetDict[key] = np.array(datasetDict[key], dtype=np.float16)

    x = [datasetDict['input_ids'], datasetDict['attention_mask'], datasetDict['tokenTypeIds']]
    y = [datasetDict['startIdxtoken'], datasetDict['endIndextoken']]
    return x, y
