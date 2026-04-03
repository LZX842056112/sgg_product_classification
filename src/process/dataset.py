from datasets import load_from_disk
from tinycss2 import tokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from configuration.config import *

# 获取数据集
def get_dataset(ds_type='train'):
    # 加载数据集
    path = str(PROCESSED_DATA_DIR / ds_type)
    dataset = load_from_disk(path)
    return dataset

# 获取数据加载器
def get_dataloader(tokenizer, ds_type='train'):
    # 加载数据集
    path = str( PROCESSED_DATA_DIR / ds_type )
    dataset = load_from_disk(path)

    # 设置格式为 tensor
    dataset.set_format(type='torch')

    # 创建DataLoader
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 获取训练数据加载器
    dataloader = get_dataloader(tokenizer)

    for batch in dataloader:
        for k, v in batch.items():
            print(k, v)
        break