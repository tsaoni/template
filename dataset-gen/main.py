from argparse import ArgumentParser, Namespace
from datasets import load_dataset

from dataset_keyphrase_extract import *
from utils import *

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="",
        default="agnews",
    )
    parser.add_argument(
        "--subdataset",
        type=str,
        help="",
        default=None,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="",
        default=None,
    )
    parser.add_argument(
        "--pad_max_len",
        type=int,
        help="",
        default=30,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="",
        default=0.0,
    )
    parser.add_argument(
        "--k_value",
        type=int,
        help="",
        default=1,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="",
        default="mask",
    )
    parser.add_argument(
        "--style_data_path",
        type=str,
        help="",
        default="fill_mask",
    )
    parser.add_argument(
        "--is_data_split",
        type=int,
        help="",
        default=1,
    )

    args = parser.parse_args()
    return args

def main(args):
    # load dataset
    if args.subdataset is None:
        dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(args.dataset, args.subdataset, cache_dir=args.cache_dir)
    print(dataset)

    # create tfidf data class
    data_mode = ['train', 'test']
    print_data_mode = ['train', 'val', 'test']
    mask_separator_token = Vocab.MASK if args.task == 'mask' else ' | '
    for mode in data_mode:
        tfidf_dataset = TfIdfDataset(dataset, mode, None, args.pad_max_len)
        print(tfidf_dataset[0])
        # mask words
        print_mode_id = [0] if mode == 'train' else [1, 2]
        for print_id in print_mode_id:
            mask_texts = mask_relevant_words(tfidf_dataset, mask_separator_token, args.task, args.threshold)
            style_transfer_args = [mask_texts, tfidf_dataset.original_text, \
                                                    tfidf_dataset.original_label]
            style_transfer_kwargs = { 'dataset type': print_data_mode[print_id],
            # World (0), Sports (1), Business (2), Sci/Tech (3).
            'label name': ['World', 'Sports', 'Business', 'SciTech'], 
            'style data path': args.style_data_path, 'is data split': args.is_data_split }
            generate_style_transfer_data(*style_transfer_args, **style_transfer_kwargs)



if __name__ == '__main__':
    args = parse_args()
    main(args)