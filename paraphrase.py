from dotenv import load_dotenv
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


load_dotenv()
hf_token = os.getenv('HF_TOKEN')

model_name = 's-nlp/rubert-base-cased-conversational-paraphrase-v1'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token
)

model = AutoModel.from_pretrained(
    model_name,
    token=hf_token
)

if torch.cuda.is_available():
    model.cuda()

model.eval()

def compare_texts_paraphrase(text1, text2):
    encoded = tokenizer(
        [text1, text2], 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    
    if torch.cuda.is_available():
        encoded = {k: v.cuda() for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :]
        similarity = F.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        )
    
    return similarity.item()