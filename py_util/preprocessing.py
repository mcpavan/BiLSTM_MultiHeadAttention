from collections import Counter
import pandas as pd
import nltk
import numpy as np
from typing import Optional, Iterable
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from py_util.embeddings_preprocessing import clean_text
from tqdm import tqdm_notebook

class PreProcess():

    def __init__(self, data: [list, pd.Series, np.ndarray], targets: [list, pd.Series, np.ndarray],
                 seq_length: Optional[int] = None, class_order: Optional[Iterable] = None, load_dir: Optional[str] = None,
                 save_dir: Optional[str] = None, char_ngram: Optional[Iterable] = [], pretr_emb_path: Optional[str] = False):
        """
        Initiates the object and preprocess the data
        :param data: 1-dimension set of instances of texts.
        :param targets: 1-dimension set of targets for each instance.
        :param seq_length: number of last words considered on each sentence.
        :param class_order: sort the target labels using this list.
        :param load_dir: string path to the pickle file containing the objects to be loaded.
        :param save_dir: string path to the pickle file where the objects will be saved.
        :param char_ngram: list to indicate the use of char n-grams, first position is the
                           min and the second position is the max for the ngram range.
                           A empty list indicates not to use it.
        :param pretr_emb_path: string containing the path where the trained
                               embeddings weights are located.
        """
        assert isinstance(char_ngram, list), "Please provide a list value for char_ngram parameter" 
        bool_ngram = isinstance(char_ngram, list) and len(char_ngram) > 0

        check_data_target(data, targets)

        self.class_order = class_order
            
        self.loaded = False
        if load_dir:
            self.loaded = True
            self.unpickle_preproc(f"{load_dir}/pre_process{'_ngram' if bool_ngram else ''}.pkl")
            save_dir = False

            if not bool_ngram:
                self.vec2tgt = {v:k for k,v in self.tgt2vec.items()}
                self.int_to_vocab = {v:k for k,v in self.vocab_to_int.items()}
                tokenized_data = self.get_tokenized_sentences(data)
        else:
            self.seq_length = seq_length
            self.pretr_emb_path = pretr_emb_path
            self.tgt2vec, self.vec2tgt, self.tgt_cnt = create_tgt_lookup_tables(targets, class_order)

            if bool_ngram:
                self.tfidfvect, self.features = get_ngram_features(char_ngram, data)
                self.vocab_to_int = self.int_to_vocab = self.word_counter = None
            else:
                assert isinstance(self.seq_length, int), "Please provide a value for seq_length parameter"
                tokenized_data = self.get_tokenized_sentences(data)
                self.tfidfvect = None
                
        if not bool_ngram:
            int_text = [self.convert_text_to_int(tokenized_instance) for tokenized_instance in tokenized_data]
            self.features = [left_pad(int_instance, self.seq_length) for int_instance in int_text]

        self.vec_targets = [self.convert_lbl_to_out(tgt) for tgt in targets]

        if save_dir:
            self.pickle_preproc(f"{save_dir}/pre_process{'_ngram' if bool_ngram else ''}.pkl")

    def convert_int_to_text(self, text):
        """
        Convert each integer representing a word of a list to an actual word
        :param text: List of integers representing words from split text
        :return: A new list of actual words
        """
        if isinstance(text, str):
            # return self.int_to_vocab[text]
            text = [text]
        
        new_text = []
        for token in text:
            new_text += [self.int_to_vocab.get(token) or ""]
        return new_text
        
    def convert_text_to_int(self, text):
        """
        Convert each word of a text to an integer
        :param text: List of words from split text
        :return: A new list of integers representing the words
        """
        if isinstance(text, str):
            # return self.vocab_to_int[text]
            text = [text]
        
        new_text = []
        for token in text:
            new_text += [self.vocab_to_int.get(token) or -1]
        return new_text

    def convert_out_to_lbl(self, tgt):
        """
        Convert each vector representing a target to an actual label
        :param tgt: List of vectors representing labels
        :return: A new list of actual labels
        """
        assert isinstance(tgt, tuple) or isinstance(tgt, float), "Target type is not tuple or float"
        if isinstance(tgt, float)==1:
            tgt = tgt[0] * len(self.tgt_cnt.keys())
        return self.vec2tgt.get(tgt) or list(self.vec2tgt.values())[0]
            
    def convert_lbl_to_out(self, tgt):
        """
        Convert a target to a vector
        :param tgt: List of targets
        :return: A new list of vectors representing the labels
        """
        assert isinstance(tgt, str), "Target is not String"
        return self.tgt2vec.get(tgt)

    def get_tokenized_sentences(self, data):
        if self.pretr_emb_path:
            tokenized_data = [tokenize_text(clean_text(data_instance)) for data_instance in tqdm_notebook(data, desc="Cleaning text", leave=False)]

            unpickled_dict = self.unpickle_vocab_wrdcnt()
            self.int_to_vocab = unpickled_dict.get("int_to_vocab")
            self.word_counter = unpickled_dict.get("word_counter")
            self.vocab_to_int = {v:k for k,v in self.int_to_vocab.items()}
        else:
            tokenized_data = [tokenize_text(data_instance) for data_instance in data]

            vocab = []
            for sentence in tqdm_notebook(tokenized_data, desc="Gerating token list", leave=False):
                vocab.extend(sentence)
            
            if not self.loaded:
                self.vocab_to_int, self.int_to_vocab, self.word_counter = create_lookup_tables(vocab)

        return tokenized_data

    def pickle_preproc(self, path):
        params = {"tgt2vec": self.tgt2vec,
                  "vocab_to_int": self.vocab_to_int,
                  "word_counter": self.word_counter,
                  "tgt_cnt": self.tgt_cnt,
                  "seq_length": self.seq_length,
                  "tfidfvect": self.tfidfvect,
                  "pretrained_emb": self.pretr_emb_path}
        
        pickle.dump(params, open(path, "wb"))

    def unpickle_preproc(self, path):
        loaded_obj = pickle.load(open(path, "rb"))

        self.tfidfvect = None
        self.pretr_emb_path = None
        if isinstance(loaded_obj, tuple): #modelo antigo de pickle, parâmetros estão em ordem (mantido para compatibilidade)
            self.tgt2vec, self.vocab_to_int, self.word_counter, self.tgt_cnt, self.seq_length = loaded_obj
        elif isinstance(loaded_obj, dict): #modelo novo de pickle, parâmetros estão dentro de um dict
            self.tgt2vec = loaded_obj.get("tgt2vec")
            self.vocab_to_int = loaded_obj.get("vocab_to_int")
            self.word_counter = loaded_obj.get("word_counter")
            self.tgt_cnt = loaded_obj.get("tgt_cnt")
            self.seq_length = loaded_obj.get("seq_length")
            self.tfidfvect = loaded_obj.get("tfidfvect")
            self.pretr_emb_path = loaded_obj.get("pretrained_emb")

    def unpickle_weights(self):
        weights=None
        
        if self.pretr_emb_path:
            weights = pickle.load(open(f"{self.pretr_emb_path}/emb_weights.pkl", "rb"))

        return weights
    
    def has_pre_weights(self):
        return self.pretr_emb_path

    def unpickle_vocab_wrdcnt(self):
        return pickle.load(open(f"{self.pretr_emb_path}/vocab_counter.pkl", "rb"))

def check_data_target(data, targets):
    #check data type
    if isinstance(data, pd.Series):
        data = list(data)
    elif isinstance(data, np.ndarray):
        data = data.to_list()
    elif not isinstance(data, list):
        tp = data.type()
        raise ValueError(f"Data type {tp} is not supported. Please provide a list, pd.Series or a np.ndarray")

    #check target type
    if isinstance(targets, pd.Series):
        targets = list(targets)
    elif isinstance(targets, np.ndarray):
        targets = targets.to_list()
    elif not isinstance(targets, list):
        tp = targets.type()
        raise ValueError(f"Targets type {tp} is not supported. Please provide a list, pd.Series or a np.ndarray")
    
    #check if the sizes of data and target are the equal
    assert len(data) == len(targets), "Data and Targets does not have the same lenght"

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: List of words from split text
    :return: A tuple of dicts (vocab_to_int, int_to_vocab, wrd_cnt)
    """
    wrd_cnt = Counter(text)
    sort_voc = sorted(wrd_cnt, key=wrd_cnt.get, reverse=True)
    int_to_vocab = {}
    vocab_to_int = {}

    for k, wrd in enumerate(sort_voc):
        int_to_vocab[k+1]=wrd
        vocab_to_int[wrd]=k+1

    return vocab_to_int, int_to_vocab, wrd_cnt

def create_tgt_lookup_tables(targets, class_order):
    """
    Create lookup tables for targets
    :param text: List of targets for each instance
    :return: A tuple of dicts (tgt2vec, vec2tgt, tgt_cnt)
    """
    tgt_cnt = Counter(targets)
    sort_voc = sorted(tgt_cnt, key=tgt_cnt.get, reverse=True)
    vec2tgt = {}
    tgt2vec = {}

    assert not class_order or set(class_order) == set(sort_voc), "class_orders must contain all classes in dataset"

    #This normalization term will be used to scale the outputs between 0 and 1
    norm_term = len(set(sort_voc))

    if class_order:
        for k, wrd in enumerate(class_order):
            vec2tgt[(k,)]=wrd
            tgt2vec[wrd]=(k/norm_term,)
    elif len(tgt_cnt)==2:
        for k, wrd in enumerate(sort_voc):
            vec2tgt[(k,)]=wrd
            tgt2vec[wrd]=(k,)
    else:
        for k, wrd in enumerate(sort_voc):
            vec = [0] * len(tgt_cnt)
            vec[k] = 1
            vec2tgt[tuple(vec)]=wrd
            tgt2vec[wrd]=tuple(vec)

    return tgt2vec, vec2tgt, tgt_cnt

def get_ngram_features(char_ngram, data, tfidfvect=None):
        if not tfidfvect:
            if len(char_ngram) < 2:
                char_ngram = [2, char_ngram[0]]
            else:
                char_ngram = list(char_ngram)[0:2]

            tfidfvect = TfidfVectorizer(analyzer="char", ngram_range=char_ngram, encoding="ISO-8859-1")
        features = tfidfvect.fit_transform(data)

        return tfidfvect, features

def left_pad(int_text, seq_length):
    if len(int_text) >= seq_length:
        return int_text[-seq_length:]
    else:
        comp = [0] * (seq_length-len(int_text))
        return comp + int_text

def tokenize_text(text, language="Portuguese"):
    """
    Tokenize the raw text
    :param text: Raw text to be tokenized
    :param language: The text language
    :return: A list of tokenized words
    """
    try:
        tokenized_text = nltk.word_tokenize(text, language=language)
    except LookupError:
        nltk.download("punkt")
        tokenized_text = nltk.word_tokenize(text, language=language)
    return tokenized_text
