from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


load_dotenv()
hf_token = os.getenv('HF_TOKEN')

model_name = 'cointegrated/rubert-base-cased-nli-threeway'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    token=hf_token
)

if torch.cuda.is_available():
    model.cuda()

def compare_texts_nli(text1, text2):
    batch = tokenizer(text1, text2, return_tensors='pt')
    if torch.cuda.is_available():
        batch = {k: v.cuda() for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
        proba = torch.softmax(outputs.logits, -1).cpu().numpy()[0]
    
    return proba[0]