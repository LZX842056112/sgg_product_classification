import time
from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from configuration.config import *
from process.dataset import get_dataset


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-5
    save_steps: int = 10
    output_dir: str = './models'
    log_dir: str = './logs'


# 训练器类
class Trainer:
    # 初始化
    def __init__(self, model, train_dataset, collate_fn, device, train_config=None):
        # 训练参数配置
        self.train_config = train_config
        # 模型和设备
        self.model = model.to(device)
        self.device = device
        # 数据集和数据整理函数
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=self.train_config.learning_rate)

        # 全局的迭代次数（运行step数）
        self.step = 1
        # Tensorboard写入器
        self.writer = SummaryWriter(log_dir=str(Path(self.train_config.log_dir) / time.strftime("%Y-%m-%d-%H-%M-%S")))

    # 定义内部方法：获取数据加载器
    def _get_dataloader(self, dataset):
        # 设置格式为 tensor
        dataset.set_format(type='torch')

        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn)

        return dataloader

    def train(self):
        self.model.train()
        # 获取训练集加载器
        dataloader = self._get_dataloader(self.train_dataset)

        # 双重for循环：外层遍历所有的epoch
        for epoch in range(self.train_config.epochs):
            # 内层循环遍历一个epoch中的所有批次
            for inputs in tqdm(dataloader, desc=f'[Epoch: {epoch + 1}]'):
                # 调用一步（step）的训练过程，得到损失
                this_loss = self._train_one_step(inputs)

                # 判断如果达到了save_steps，就记录损失、判断是否保存模型
                if self.step % self.train_config.save_steps == 0:
                    # 记录loss
                    tqdm.write(f'[Epoch:{epoch + 1} | Step:{self.step}]  Loss: {this_loss}')
                    self.writer.add_scalar('loss', this_loss, self.step)

                    # 判断是否保存模型
                    if this_loss < self.min_loss:
                        self.min_loss = this_loss
                        tqdm.write("保存模型...")
                        self.model.save_pretrained(self.train_config.output_dir)

                    self.step += 1  # 迭代次数加1

    # 一步训练（一次迭代）
    def _train_one_step(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 前向传播
        outputs = self.model(**inputs)
        loss = outputs.loss
        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()


if __name__ == '__main__':
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 3. 加载 labels
    with open(MODEL_DIR / LABELS_FILE, 'r', encoding='utf-8') as f:
        all_labels = f.read().splitlines()
    # 转成label和id的映射字典
    id2label = {index: label for index, label in enumerate(all_labels)}
    label2id = {label: index for index, label in enumerate(all_labels)}

    # 4. 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
    )
    # print(model.config.id2label)
    # model.save_pretrained(MODEL_DIR)

    train_dataset = get_dataset('train')
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )

    # 7. 定义训练配置
    train_config = TrainConfig(batch_size=16, output_dir=MODEL_DIR, log_dir=LOG_DIR)

    # 8. 定义训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        collate_fn=collate_fn,
        device=device,
        train_config=train_config
    )

    # 9. 训练
    trainer.train()
