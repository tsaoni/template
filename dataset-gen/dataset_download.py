import os
from argparse import ArgumentParser, Namespace
from datasets import load_dataset


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="glue",
    )
    args = parser.parse_args()
    return args

def main(args):
    data_path = '~/.cache/huggingface/datasets/'
    # if not os.path.isdir(data_path):
    #     os.mkdir(data_path)
    if args.dataset == 'glue':
        glue_config = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                    'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        for i, config in enumerate(glue_config):
            dataset = load_dataset('glue', config)
            print(dataset)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
