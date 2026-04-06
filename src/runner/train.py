import time
from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler
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
    save_steps: int = 100
    output_dir: str = './models'
    log_dir: str = './logs'
    early_stop_metric: str = 'loss'  # 早停指标（loss, acc, f1）
    early_stop_patience: int = 5  # 容忍度
    use_amp: bool = True  # 是否开启AMP


# 训练器类
class Trainer:
    # 初始化
    def __init__(self, model, valid_dataset, collate_fn, compute_metrics, device, train_dataset=None,
                 train_config=TrainConfig()):
        # 训练参数配置
        self.train_config = train_config
        # 模型和设备
        self.model = model.to(device)
        self.device = device
        # 数据集和数据整理函数
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collate_fn = collate_fn
        # 评估函数
        self.compute_metrics = compute_metrics
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=self.train_config.learning_rate)

        # 全局的迭代次数（运行step数）
        self.step = 1
        # Tensorboard写入器
        self.writer = SummaryWriter(log_dir=str(Path(self.train_config.log_dir) / time.strftime("%Y-%m-%d-%H-%M-%S")))
        # 全局最佳评估得分
        self.early_stop_best_score = -float("inf")
        # 容忍度计数器
        self.early_stop_counter = 0

        # AMP 梯度缩放器
        self.scaler = GradScaler(device=self.device.type, enabled=self.train_config.use_amp)

        # 检查点文件路径
        self.checkpoint_path = Path(self.train_config.output_dir) / 'last' / 'checkpoint.pt'

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

    # 核心方法
    def train(self):
        # 加载检查点
        self._load_checkpoint()

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

                    # 验证，得到验证指标
                    metrics = self.evaluate()
                    metrics_str = '|'.join([f'{k}:{v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f'[Evaluate: {metrics_str}]')

                    # 早停判断处理
                    if self._should_stop(metrics):
                        tqdm.write('早停')
                        return

                    # 保存检查点
                    self._save_checkpoint()

                self.step += 1  # 迭代次数加1

    # 一步训练（一次迭代）
    def _train_one_step(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 前向传播，增加autocast上下文
        with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.train_config.use_amp
        ):
            outputs = self.model(**inputs)
            loss = outputs.loss
        # 反向传播，增加GradScaler操作
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    # 核心验证方法，返回一个字典，记录不同的评价指标：{loss:0.35, acc:0.89, f1:0.76}
    def evaluate(self) -> dict:
        # 获取验证集数据加载器
        dataloader = self._get_dataloader(self.valid_dataset)
        self.model.eval()

        total_loss = 0.0
        all_labels = []  # 所有数据真实标签
        all_preds = []  # 所有数据预测标签

        for inputs in tqdm(dataloader, desc=f'[Evaluate]'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # 前向传播
            outputs = self.model(**inputs)
            # 获取损失
            loss = outputs.loss
            total_loss += loss.item()
            # 预测分类结果
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.tolist())
            # 获取标签合并到列表
            labels = inputs['labels']
            all_labels.extend(labels.tolist())
        # 遍历完验证集，计算平均损失和其它指标
        loss = total_loss / len(dataloader)
        metrics = self.compute_metrics(all_preds, all_labels)  # 返回一组指标的字典
        return {'loss': loss, **metrics}

    def _should_stop(self, metrics):
        # 提取配置项中定义的指标值
        metric = metrics[self.train_config.early_stop_metric]
        # 转换评分
        score = -metric if self.train_config.early_stop_metric == 'loss' else metric
        # 判断如果超过最佳得分，就保存
        if score > self.early_stop_best_score:
            self.early_stop_best_score = score
            self.early_stop_counter = 0  # 计数清零
            tqdm.write('保存最佳模型...')
            self.model.save_pretrained(str(Path(self.train_config.output_dir) / "best"))
            return False
        else:
            self.early_stop_counter += 1  # 计数加1
            # 如果达到上限，就早停
            if self.early_stop_counter >= self.train_config.early_stop_patience:
                return True
            else:
                return False

    # 保存检查点函数
    def _save_checkpoint(self):
        # 定义字典，用于保存检查点
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'step': self.step,  # 迭代次数
            'early_stop_best_score': self.early_stop_best_score,
            'early_stop_counter': self.early_stop_counter,
        }
        # 保存到文件
        torch.save(checkpoint, self.checkpoint_path)

    # 加载检查点函数
    def _load_checkpoint(self):
        # 判断如果检查点路径存在，就加载
        if self.checkpoint_path.exists():
            tqdm.write("发现检查点，加载状态继续训练...")
            # 获取检查点字典
            checkpoint = torch.load(self.checkpoint_path)
            # 对状态赋值
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.step = checkpoint['step']
            self.early_stop_best_score = checkpoint['early_stop_best_score']
            self.early_stop_counter = checkpoint['early_stop_counter']
        else:
            tqdm.write("没有检测到检查点，从头开始训练...")


def train():
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

    # 5. 数据集和整理函数
    train_dataset = get_dataset('train')
    valid_dataset = get_dataset('valid')
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )

    # 6. 评估函数：根据实际需求定义，acc 和 f1
    def compute_metrics(preds, labels) -> dict:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {'acc': acc, 'f1': f1}

    # 7. 定义训练配置
    train_config = TrainConfig(batch_size=16, output_dir=MODEL_DIR, log_dir=LOG_DIR)

    # 8. 定义训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=collate_fn,
        compute_metrics=compute_metrics,
        device=device,
        train_config=train_config
    )

    # 9. 训练
    trainer.train()


# 训练调用流程
if __name__ == '__main__':
    train()
