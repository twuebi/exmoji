import sys
from processing.loader import Set
from nn.train import train

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s TRAIN_SET DEV_SET\n" % sys.argv[0])
        #sys.exit(1)

    train_data = Set(train=True)
    train_data.load(sys.argv[1])

    test_data = Set(train=False)
    test_data.load(sys.argv[2])

    train(train_data,test_data)