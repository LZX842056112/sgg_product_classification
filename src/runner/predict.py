# 预测器类
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configuration.config import *


class Predictor:
    # 初始化，传入model，tokenizer，device
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    # 核心预测方法
    def predict(self, texts: str | list):
        # 统一数据格式：如果是字符串，转换成列表
        is_str = isinstance(texts, str)
        if is_str:
            texts = [texts]

        # 1. 分词编码，得到模型输入
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 2. 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 3. 根据输出结果，解码得到中文分类标签
        preds = torch.argmax(outputs.logits, dim=-1).tolist()
        labels = [self.model.config.id2label[pred_id] for pred_id in preds]

        if is_str:
            return labels[0]
        return labels


# 测试主流程
def predict():
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    # 3. 加载微调后的模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "best")
    # 4. 创建预测器
    predictor = Predictor(model, tokenizer, device)
    # 5. 预测
    text = "好奇心钻装纸尿裤L40片9-14kg"
    result = predictor.predict(text)
    print(result)
    texts = ["240ML*15养元2430六个核桃", "潘婷丝质顺滑洗发露750ml", "640G正航牛奶早餐饼干"]
    result = predictor.predict(texts)
    print(result)


if __name__ == '__main__':
    predict()
