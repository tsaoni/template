import numpy as np
from torch.utils.data import Dataset
from utils import Vocab
from itertools import chain
from nltk.tokenize import word_tokenize

import pdb

from utils import custom_word_tokenize

class TfIdfDataset(Dataset):
    def __init__(self, dataset, mode, vocab_file, pad_max_len):
        self.original_text = []
        self.tokenize_text = []
        self.text_TfIdf = []
        self.original_label = []
        self.pad_max_len = pad_max_len
        self.vocab = Vocab(vocab_file)
        self.data_num = len(dataset) if mode is None else len(dataset[mode])

        self._document_num = {}
        self._word_count = {}

        if mode == None:
            self.preprocess(dataset)
        else:
            self.preprocess(dataset[mode])

        

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        word_tfidf = []
        def make_word_tfidf_pair():
            nonlocal word_tfidf
            word_exist = []
            for word, tfidf in zip(custom_word_tokenize(self.original_text[idx], True), self.text_TfIdf[idx]):
                if word not in word_exist:
                    word_exist.append(word)
                    word_tfidf.append((word, tfidf))

        make_word_tfidf_pair()
        word_tfidf.sort(key=lambda x: x[1], reverse=True)
        sorted_text = [word[0] for word in word_tfidf]
        sorted_tfidf = [word[1] for word in word_tfidf]

        return {'original text': self.original_text[idx], 'original label': self.original_label[idx], 
                'sorted text': sorted_text,
                #'mask text': ret_mask, 'mask label': mask_label[idx], 
                'sorted tfidf': sorted_tfidf,
                'original tfidf': self.text_TfIdf[idx] }

    def split_original_text(self):
        split_texts = []
        for text in self.original_text:
            split_texts.append(custom_word_tokenize(text, False))
        
        return split_texts

    def preprocess(self, dataset):
        for data in dataset:
            self.original_text.append(data['text'])
            self.original_label.append(data['label'])
        self.tokenize_text = self.tokenize(self.original_text)
        self.text_TfIdf = self.tfidf(self.tokenize_text, self.original_label)


    def tokenize(self, texts):
        tokenize_text = []
        for text in texts:
            words = custom_word_tokenize(text, True)
            tokenize_text.append(words)
        return self.vocab.encode_batch(tokenize_text, self.pad_max_len)

    # term frequency
    def termfreq(self, document_id, target_token):
        return self._word_count[str(target_token)][document_id] / self._document_num[document_id]

    # inverse document frequency
    def inverse_doc_freq(self, target_token):
        # return np.log(self._document_num['total'] / self._word_count[str(target_token)]['total'])
        return self._document_num['total'] / self._word_count[str(target_token)]['total']

    def tfidf(self, tokenize_text, label):
        total_tfidf = []
        self.calculate_document()
        self.word_count()
        for i, tokens in enumerate(tokenize_text):
            text_tfidf = []
            for token in tokens:
                # if the word is PAD, skip it
                if token == self.vocab.PAD:
                    text_tfidf.append(-1)
                else:
                    tf = self.termfreq(str(self.original_label[i]), token)
                    idf = self.inverse_doc_freq(token)
                    tfidf = tf * idf
                    text_tfidf.append(tfidf)
            total_tfidf.append(text_tfidf)  

        return total_tfidf      

    def calculate_document(self):
        self._document_num = {'total': 0, '0': 0, '1': 0, '2': 0, '3': 0}
        for label in self.original_label:
            self._document_num['total'] += 1
            self._document_num[str(label)] += 1

    def word_count(self):
        token_word_set = set(chain.from_iterable(self.tokenize_text))
        for token_word in token_word_set:
            self._word_count[str(token_word)] = {'total': 0, '0': 0, '1': 0, '2': 0, '3': 0}
        for i, tokens in enumerate(self.tokenize_text):
            for token in set(tokens):
                self._word_count[str(token)]['total'] += 1
                self._word_count[str(token)][str(self.original_label[i])] += 1
        
def delete_relevant_words(dataset, threshold):
    mask_texts = []
    for data_object in dataset:
        mask_text = delete_relevant_words_in_object(data_object, threshold)
        mask_texts.append(mask_text)
    
    return mask_texts

def delete_relevant_words_in_object(data_object, threshold):
    original_text_list = custom_word_tokenize(data_object['original text'])
    delete_word = []
    for word, tfidf in zip(data_object['sorted text'], data_object['sorted tfidf']):
        if tfidf > threshold:
            delete_word.append(word)
        else:
            break
    for i, text in enumerate(original_text_list):
        if text.lower() in delete_word:
            original_text_list = list(filter((text).__ne__, original_text_list)) 

    """
    if count < 10:
        count += 1
        print('original: ', data_object['original text'])
        print('delete: ', delete_word)
        print('mask: ', " ".join(original_text_list))
    """

    return " ".join(original_text_list)

def mask_relevant_words(dataset, mask_separator_token, task, threshold):
    mask_texts = []
    for data_object in dataset:
        mask_text = mask_relevant_words_in_object(data_object, mask_separator_token, task, threshold)
        mask_texts.append(mask_text)
    
    return mask_texts

def mask_relevant_words_in_object(data_object, mask_separator_token, task, threshold):
    original_text_list = custom_word_tokenize(data_object['original text'], False)
    keyword = []
    for word, tfidf in zip(data_object['sorted text'], data_object['sorted tfidf']):
        if tfidf > threshold:
            keyword.append(word)
        else:
            break
    
    if len(keyword) == 0:
        keyword.append(data_object['sorted text'][0])
    
    mask_sentence = ""
    if task == 'mask':
        def add_text_to_mask_sentence(text):
            nonlocal mask_sentence
            if mask_sentence == "":
                mask_sentence = mask_separator_token + " " + text
            elif text == "." or text == ",":
                mask_sentence += (" " + mask_separator_token + text)
            else:
                mask_sentence += (" " + mask_separator_token + " " + text) 
    
        for i, text in enumerate(original_text_list):
            is_punctuation = (text == ',' or text == '.')
            if text.lower() in keyword or is_punctuation:
                add_text_to_mask_sentence(text)

    elif task == 'keyword':
        mask_sentence = mask_separator_token.join(keyword)

    return Vocab.MASK if mask_sentence == "" else mask_sentence