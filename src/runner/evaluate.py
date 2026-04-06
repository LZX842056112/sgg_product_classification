import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from configuration.config import *
from process.dataset import get_dataset
from runner.train import Trainer, TrainConfig


# 验证流程
def evaluate():
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 3. 加载微调后的模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / 'best')

    # 4. 数据集和整理函数
    test_dataset = get_dataset('train')
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )

    # 5. 评估函数：根据实际需求定义，acc 和 f1
    def compute_metrics(preds, labels) -> dict:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {'acc': acc, 'f1': f1}

    # 7. 定义训练器
    trainer = Trainer(
        model=model,
        valid_dataset=test_dataset,
        collate_fn=collate_fn,
        compute_metrics=compute_metrics,
        device=device,
    )

    # 评估
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == '__main__':
    evaluate()
