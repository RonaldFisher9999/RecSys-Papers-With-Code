import argparse

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--bool', action='store_true')
        # self.parser.add_argument('--list', action='extend', nargs='+', type=int, default=[1, 2, 10])
        self.parser.add_argument('--list', nargs='+', type=int, default=[1, 2, 10])
        self.args = self.parser.parse_args()
        # self.square()

    def square(self):
        self.args.list = [i ** 2 for i in self.args.list]

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bool', action='store_true')
    parser.add_argument('--list', action='extend', nargs='+', type=int)
    # parser.add_argument('--list', nargs='+', default=[1, 2, 10])
    
    args = parser.parse_args()
    args.list = [i ** 2 for i in args.list]
    return args