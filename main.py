import argparse
from solver import Solver


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", dest="train", default = "train.txt")
    parser.add_argument("-valid", dest="valid", default = "valid.txt")
    parser.add_argument("-test", dest="test", default = "test.txt")
    args = parser.parse_args()

    solver = Solver(args.train, args.valid, args.test)
    solver.train_()

