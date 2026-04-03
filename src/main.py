from sys import argv

if __name__ == '__main__':
    print(argv)
    arg = argv[1]

    match arg:
        case 'preprocess':
            from process.preprocess import preprocess

            preprocess()
        case 'train':
            from runner.train import train

            train()
        case 'predict':
            from runner.predict import predict

            predict()
