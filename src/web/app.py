import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configuration.config import *

from runner.predict import Predictor
from web.schemas import Title, Category
from web.service import TitleService

app = FastAPI()

# 创建预测器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "best")
predictor = Predictor(model=model, tokenizer=tokenizer, device=device)

# 创建服务
service = TitleService(predictor=predictor)


@app.post("/predict")
def predict(title: Title) -> Category:
    label = service.predict(title.text)
    return Category(category=label)


def serve():
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000)
