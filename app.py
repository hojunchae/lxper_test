import torch
from transformers import AutoModelForSequenceClassification,AutoConfig,RobertaTokenizer

from fastapi import FastAPI
from pydantic import BaseModel



base_dir = "./roberta-base"

config = AutoConfig.from_pretrained(base_dir,num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(base_dir,config=config)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
softmax = torch.nn.Softmax(dim=1)

def get_result(param_string):
	return float(softmax(model(torch.tensor([tokenizer.encode(param_string,max_length=514,truncation=True,padding="max_length")]))[0])[0][1])


class userRequest(BaseModel):
	passage: str


app = FastAPI()

@app.post("/generate/")
async def generate(userrequest:userRequest):
	print(userrequest.passage)
	return {"prob":get_result(userrequest.passage)}
