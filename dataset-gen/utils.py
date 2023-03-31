import os
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize
from collections.abc import Iterable
from typing import List
from nltk.corpus import words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from collections import defaultdict
from sklearn import metrics
from sklearn.cluster import KMeans

class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"
    MASK = "<mask>"

    def __init__(self, vocab_file = None) -> None:
        if vocab_file is None:
            vocab = words.words()
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            Vocab.MASK: 2,
            **{token: i for i, token in enumerate(vocab, 3)},
        }

    # return pad id
    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    # return undefined id
    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    # return masked id
    @property
    def mask_id(self) -> int:
        return self.token2idx[Vocab.MASK]

    # return all the tokens
    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    # return token id
    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)
    
    # return tokens id
    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    # return ids
    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        # turn token to id
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        # padding
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids

def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = []
    for seq in seqs:
        if len(seq) > to_len:
            paddeds.append(seq[:to_len])
        else:
            paddeds.append(seq[:to_len] + [padding] * max(0, to_len - len(seq)))
    return paddeds

def k_means_clustering(text_list, k_value, labels, name):
    text_vector_list, lsa, vectorizer = vectorize(text_list)
    kmeans = KMeans(n_clusters=k_value, max_iter=100, n_init=1)
    new_label = fit_and_evaluate(kmeans, text_vector_list, labels, name="KMeans\nwith LSA on tf-idf vectors")
    print_top_term_per_cluster(lsa, kmeans, k_value, vectorizer, name)
    return new_label

def vectorize(text_list):
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, stop_words="english")
    text_vector_list = vectorizer.fit_transform(text_list)
    # reduce dimension
    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    text_vector_list = lsa.fit_transform(text_vector_list)

    return text_vector_list, lsa, vectorizer

def print_top_term_per_cluster(lsa, kmeans, k_value, vectorizer, name):
    original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(k_value):
        print(f"{name} Cluster {i}: ", end="")
        for ind in order_centroids[i, :10]:
            print(f"{terms[ind]} ", end="")
        print()

def fit_and_evaluate(km, X, labels, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )

    evaluation = {
        "estimator": name,
        # "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        # "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score

    return km.labels_.tolist()

def new_dataset(text_list, label_list):
    return [{'text': text, 'label': label} for text, label in zip(text_list, label_list)]

def get_correlation_matrix(data1, data2, name1, name2):
    df1, key1 = one_hot_dataframe(data1, name1)
    df2, key2 = one_hot_dataframe(data2, name2)
    df_corr = pd.concat([df1, df2], axis=1).corr()
    return df_corr.loc[key1, key2]

def one_hot_dataframe(data, name):
    df_dict = {f'{name}_0': [],f'{name}_1': [],f'{name}_2': [],f'{name}_3': []}
    keys = [f'{name}_0', f'{name}_1', f'{name}_2', f'{name}_3']
    for label in data:
        df_dict[keys[label]].append(1)
        for other in range(4):
            if other is not label:
                df_dict[keys[other]].append(0)

    return pd.DataFrame(df_dict), keys

def generate_style_transfer_data(*args, **kwargs):
    # args: source_texts, target_texts, labels
    # kwargs: dataset_type, label_name, style_data_path, is_data_split
    source_texts, target_texts, labels = args
    dataset_type = kwargs['dataset type']
    label_name = kwargs['label name']
    style_data_path = kwargs['style data path']
    is_data_split = kwargs['is data split']

    if not os.path.exists(style_data_path):
        for name in label_name:
            os.makedirs(style_data_path / Path(name))

    source_files = []
    target_files = []
    src_tgt_separator = '@separator@'
    for name in label_name:
        source_files.append(open(style_data_path / Path(name) / Path(dataset_type + '.source'), 'w'))
        if is_data_split:
            target_files.append(open(style_data_path / Path(name) / Path(dataset_type + '.target'), 'w'))
    for src_text, target_text, label in zip(source_texts, target_texts, labels):
        if is_data_split:
            source_files[label].write(src_text + '\n')
            target_files[label].write(target_text + '\n')
        else:
            src_tgt_text = src_text + ' ' + src_tgt_separator + ' ' + target_text
            source_files[label].write(src_tgt_text + '\n')

    if len(target_files) == 0:
        for src_file in source_files:
            src_file.close()
    else:
        for src_file, tgt_file in zip(source_files, target_files):
            src_file.close()
            tgt_file.close()


def custom_word_tokenize(sentence, is_lower):
    word_list = word_tokenize(sentence)
    new_word_list = []
    separator = '\\'
    for word in word_list:
        word_split = [raw_word.lower() if is_lower else raw_word \
                         for raw_word in word.split(separator)]
        new_word_list += word_split
    
    return new_word_list
