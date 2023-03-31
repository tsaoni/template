import os
import json
from pathlib import Path
from argparse import ArgumentParser, Namespace
from datasets import load_dataset, Dataset, DatasetDict

from dataset_keyphrase_extract import *
from utils import *

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="",
        default="agnews",
    )
    parser.add_argument(
        "--dataset_subname",
        type=str,
        help="",
        default=None,
    )
    parser.add_argument(
        "--data_output_dir",
        type=str,
        help="",
        default=".",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="",
        default=None,
    )
    parser.add_argument(
        "--dataset_mode",
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

class DatasetPreprocess:

    PROMPT_OPT = dict(
        nli_prompt_0=lambda sent1, sent2: "{} Is it true that {} ?".format(sent1, sent2),
        nli_prompt_1=lambda sent1, sent2: "{} @separater@ {}".format(sent1, sent2),
        text2rel_source_prompt_0=lambda source: "{} <mask> is <mask> of <mask>".format(source), 
        text2rel_target_prompt_0=lambda source, target, rel: "{} is {} of {}".format(target, rel, source), 
    )

    def __init__(self, *dataset_args, **dataset_kwargs):
        self.task = dataset_args[0]
        if '_download' in dataset_args[0]: # load dataset from file
            if 'wiki80' in dataset_args[0]:
                train_file = 'wiki80_modified_train.txt'
                val_file = 'wiki80_modified_val.txt'
                rel_file = 'rel2wiki.json'
                train_ds = load_dataset('json', data_files=Path(dataset_args[0]).joinpath(train_file).as_posix())['train']
                val_ds = load_dataset('json', data_files=Path(dataset_args[0]).joinpath(val_file).as_posix())['train']
                rel_obj = json.load(open(Path(dataset_args[0]).joinpath(rel_file).as_posix(), 'r'))
                self.rel_obj = {value: key for key, value in rel_obj.items()}
                self.dataset = DatasetDict({'train': train_ds, 'val': val_ds, 'test': val_ds, })
                
        else: # load from hf
            dataset_args = dataset_args if dataset_args[1] is not None else [dataset_args[0]]
            self.dataset = load_dataset(*dataset_args, **dataset_kwargs)
            """
            if 'val' not in self.dataset.keys():
                self.dataset['val'] = self.dataset['test']
            """
    
    def print_line_data_to_file(
        self, 
        dataset_mode, 
        data_output_path, 
        data_type_dict, 
        print_target=True, 
        prompt_opt='prompt_0'
    ):
        if not os.path.exists(data_output_path):
            Path(data_output_path).mkdir(parents=True, exist_ok=True)
        if dataset_mode == 'style-transfer':
            if self.task == 'yahoo_answers_topics':
                attrs = ['society', 'science', 'health', 'education', 
                        'computer', 'sports', 'business', 'entertainment', 'family', 'politics']
            for attr in attrs:
                os.makedirs(Path(data_output_path) / attr, exist_ok=True)

        for key, value in data_type_dict.items():
            # print lines of data into .source, .target
            if print_target:
                data_src_file = os.path.join(data_output_path, value + '.source')
                data_tgt_file = os.path.join(data_output_path, value + '.target')
                with open(data_src_file, 'w') as f1, open(data_tgt_file, 'w') as f2:
                    src_sents, tgt_sents = [], []
                    for data in self.dataset[key]:
                        if dataset_mode == 'nli':
                            src_sent = DatasetPreprocess.PROMPT_OPT[f"{dataset_mode}_" + prompt_opt]\
                                                            (data['sentence1'], data['sentence2'])
                            tgt_sent = str(data['label'])
                        elif dataset_mode == 'classification':
                            src_sent = data['text']
                            tgt_sent = str(data['label'])
                        elif dataset_mode == 'summarization':
                            src_sent = data['text']
                            tgt_sent = data['target']
                        elif dataset_mode == 'text2rel':
                            # especially for wiki80
                            sent = " ".join(data['token'])
                            src_sent = DatasetPreprocess.PROMPT_OPT[f"{dataset_mode}_source_" + prompt_opt](sent)
                            tgt_sent = DatasetPreprocess.PROMPT_OPT[f"{dataset_mode}_target_" + prompt_opt]\
                                        (data['h']['name'], data['t']['name'], self.rel_obj[data['relation']])
                            
                        src_sents.append(src_sent.strip().replace('\n', ' '))
                        tgt_sents.append(tgt_sent.strip().replace('\n', ' '))

                    f1.write('\n'.join(src_sents))
                    f2.write('\n'.join(tgt_sents))
            else: # print style transfer data
                data_files = []
                for attr in attrs:
                    data_path = os.path.join(data_output_path, attr, value)
                    data_files.append(open(data_path, 'w'))
                sents = [[] for i in range(len(attrs))]
                for data in self.dataset[key]:
                    if dataset_mode == 'style-transfer':
                        if self.task == 'yahoo_answers_topics':
                            sent = data['best_answer']
                            ind = data['topic']

                    sents[ind].append(sent.strip().replace('\n', ' '))

                for i in range(len(attrs)):
                    data_files[i].write('\n'.join(sents[i]))
                                

    @staticmethod
    def check_hf_dataset_split(task):
        classification_task = ['imdb', 'yelp_polarity', ]
        # key: dataset split name, value: dataset file name
        if task == 'mrpc':
            data_split_dict = dict(
                train='train',
                validation='val',
                test='test',
            )
        elif task == 'news-summary':
            data_split_dict = dict(
                train='train',
                test='test',
            )
        elif task == 'wiki80':
            data_split_dict = dict(
                train='train',
                val='val',
                test='test',
            )
        elif task == 'yahoo_answers_topics':
            data_split_dict = dict(
                train='train',
                val='dev',
                test='test',
            )
        elif task == 'rotten_tomatoes':
            data_split_dict = dict(
                train='train',
                validation='val',
                test='test',
            )
        elif task in classification_task:
            data_split_dict = dict(
                train='train',
                val='val',
                test='test',
            )

        return data_split_dict

def check_variable_status(variable, name="", status='None'):
    if status == 'None':
        if variable is None:
            raise ValueError('{} parameter should not be none. '.format(name))

def main(args):
    # load dataset process object
    check_variable_status(args.dataset_name_or_path, name='dataset_name_or_path')
    dataset_process_object = DatasetPreprocess(args.dataset_name_or_path, args.dataset_subname, \
                                                                    cache_dir=args.cache_dir)
    if args.dataset_subname is not None:
        data_type_dict_name = args.dataset_subname
    else:
        data_type_dict_name = args.dataset_name_or_path
        # data_type_dict_name = args.dataset_name_or_path.split('/')[-1].split('_')[0]
    data_type_dict = DatasetPreprocess.check_hf_dataset_split(data_type_dict_name)
    dataset_process_object.print_line_data_to_file(args.dataset_mode, args.data_output_dir, \
                                                        data_type_dict)
    
    """
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
    """


if __name__ == '__main__':
    args = parse_args()
    main(args)