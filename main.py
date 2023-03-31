# from argparse import ArgumentParser, Namespace

from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

@dataclass
class DataTrainingArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

""" if not using huggingface arguments
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="",
        default="agnews",
    )
    args = parser.parse_args()
    return args
"""

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    

if __name__ == '__main__':
    # args = parse_args()
    main()
