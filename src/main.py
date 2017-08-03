import sys
from exmoji import Datalist
from exmoji import train, train_sparse

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("USAGE: {} train_set dev_set\n".format(sys.argv[0]))
        sys.exit(1)

    train_data = Datalist()
    train_data.load(sys.argv[1])

    test_data = Datalist(train_data.numberers)
    test_data.load(sys.argv[2])
    train(train_data,test_data)
