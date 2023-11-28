from config import args

def main():
    for k, v in vars(args).items():
        print(k, v)

if __name__=='__main__':
    main()