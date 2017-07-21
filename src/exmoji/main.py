import sys
from processing.loader import Data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s TRAIN_SET DEV_SET\n" % sys.argv[0])
        #sys.exit(1)

    train_data = Data(train=True)
    train_data.load(sys.argv[1])

    test_data = Data(train=False)
    test_data.load(sys.argv[2])