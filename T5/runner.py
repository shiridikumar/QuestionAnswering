import nltk
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from colorama import  Back, Style,Fore
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
import torch
import warnings

# Filter warnings by category
warnings.filterwarnings("ignore")
model.load_state_dict(torch.load("fine_tuned_T5.pth",map_location="cpu"))
def predict_answer(context,question):
  inputs=tokenizer(
            context,
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512, 
        )
  with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model.generate(**inputs)
    return output

from colorama import  Back, Style
while(1):
    context=input(Fore.BLUE+"Context : " )
    question=input(Fore.RED+"Question : ")
    ans=predict_answer(context,question)
    ans=tokenizer.decode(ans[0], skip_special_tokens=True)
    print(Fore.GREEN+"Answer  : "+ans )
    print()
