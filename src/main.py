from argparse import ArgumentParser

if __name__ == '__main__':
    # 定义一个参数解析器
    parser = ArgumentParser(usage='usage: main.py action')
    # 添加具体参数，指定候选项
    parser.add_argument('action', choices=['train', 'predict', 'evaluate', 'preprocess', 'serve'])
    # 解析参数
    args = parser.parse_args()
    action = args.action

    match action:
        case 'preprocess':
            from process.preprocess import preprocess

            preprocess()
        case 'train':
            from runner.train import train

            train()
        case 'predict':
            from runner.predict import predict

            predict()
        case 'evaluate':
            from runner.evaluate import evaluate

            evaluate()
        case 'serve':
            from web.app import serve

            serve()
