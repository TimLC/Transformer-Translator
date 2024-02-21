import nltk
from nltk import WordNetLemmatizer, TreebankWordTokenizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import spacy
from collections import Counter, defaultdict
import enchant

import re
import os
import pickle

from spacy.attrs import ORTH

from tokenizer.referential import *


class CustomTokenizer:
    def __init__(
            self,
            lower: bool = True,
            start_token: str = '<sos>',
            end_token: str = '<eos>',
            num_token: str = '<num>',
            noun_token: str = '<noun>',
            empty_token: str = '<pad>',
            unknown_token: str = '<unk>',
            filters: list = ['"', '#', '/', '(', ')', '[', '\\', ']', '^', '_', '`', '´', '{', '|', '}', '~', '\t', '\n'],
            language='english'
    ) -> None:
        self.unknown_token = unknown_token
        self.lower = lower
        self.start_token: str = start_token
        self.end_token: str = end_token
        self.num_token: str = num_token
        self.noun_token: str = noun_token
        self.empty_token: str = empty_token
        self.filters: list = filters
        self.language: str = language
        self.sentences: list[list[str]] = []
        self.word_to_index: dict = {}
        self.index_to_word: dict = {}
        self.max_length: int = 0
        self.number_regex: str = '[0-9]+([.|,][0-9]+)?'
        self.PATH_DATASET = 'resources/tokenizer'

        if self.language == 'english':
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            self.suffix_noun = SUFFIX_NOUN_EN
            self.suffix_ver = SUFFIX_VER_EN
            self.suffix_adj = SUFFIX_ADJ_EN
            self.suffix_adv = SUFFIX_ADV_EN
        elif self.language == 'french':
            self.nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
            self.suffix_noun = SUFFIX_NOUN_FR
            self.suffix_ver = SUFFIX_VER_FR
            self.suffix_adj = SUFFIX_ADJ_FR
            self.suffix_adv = SUFFIX_ADV_FR

        special_case = [{ORTH: self.unknown_token}]
        self.nlp.tokenizer.add_special_case(self.unknown_token, special_case)

    def tokenizer(self, text: list[str]) -> None:

        for index, sentence in enumerate(text):
            # if 'νατο' in sentence:
            #     print(sentence)
            text[index] = self.filter_sentence(sentence)

        docs = self.nlp.pipe(text, n_process=4, batch_size=1000)
        for sentence in docs:
            list_token = []
            for token in sentence:
                list_token.extend(self.transformer_token(token))
            list_token = [self.start_token] + list_token + [self.end_token]
            self.max_length = max(self.max_length, len(list_token))
            self.sentences.append(list_token)

        all_token_found = [token for sentence in self.sentences for token in sentence]
        freq_token = Counter(all_token_found)
        tokens = [k for k, v in freq_token.items() if v > 1]

        for index_sentence, sentence in enumerate(self.sentences):
            for index_token, token in enumerate(sentence):
                if not token in tokens:
                    self.sentences[index_sentence][index_token] = self.unknown_token

        if not self.unknown_token in tokens:
            tokens.append(self.unknown_token)
        tokens.append(self.empty_token)

        self.index_to_word = dict((k, v) for k, v in enumerate(tokens))
        self.word_to_index = dict((v, k) for k, v in self.index_to_word.items())

        with open('items-en.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(self.index_to_word.values()))

        sentences = []
        for sentence in self.sentences:
            sentence = [self.word_to_index[word] for word in sentence]
            sentences.append(sentence)
        self.sentences = sentences

    def filter_sentence(self, sentence):
        sentence = ''.join(x for x in sentence if not x in ['<', '>'])
        sentence = re.sub(r'https?://\S+', self.unknown_token, sentence)
        sentence = ''.join(x for x in sentence if not x in self.filters)
        sentence = sentence.replace('\'', ' ').replace('/', ' ').replace('-', ' ').replace('­', ' ')
        sentence = re.sub('\s+', ' ', sentence)
        return sentence

    def transformer_token(self, token):
        text_token = token.text.lower()

        if bool(re.search(r'\d', text_token)):
            return [self.unknown_token]
        elif bool(re.search(r'(\w+\.)|(\.\w+)', text_token)):
            return [self.unknown_token]
        elif token.pos_ == 'PROPN':
            return [self.noun_token]
        elif token.pos_ == 'NUM':
            return [self.num_token]
        elif token.pos_ == 'NOUN':
            return self.split_token(text_token, self.suffix_noun)
        elif token.pos_ == 'VERB':
            return self.split_token(text_token, self.suffix_ver)
        elif token.pos_ == 'ADJ':
            return self.split_token(text_token, self.suffix_adj)
        elif token.pos_ == 'ADV':
            return self.split_token(text_token, self.suffix_adv)

        else:
            return [text_token]

    def split_token(self, token, suffixes):
        found = False
        for s in suffixes:
            if token.endswith(s):
                token = [token[: len(token) - len(s)], '_' + token[len(token) - len(s):]]
                found = True
                break
        if found:
            return token
        else:
            return [token]

    def tokenize(self, sentence: str) -> list[int]:
        if self.lower:
            sentence = sentence.lower()
        sentence = self.filter_sentence(sentence)
        sentence = self.nlp(sentence)
        sentence = [self.start_token] + sentence + [self.end_token]

        for index_token, token in enumerate(sentence):
            if not token in self.word_to_index.keys():
                sentence[index_token] = self.unknown_token

        return [self.word_to_index[word] for word in sentence]

    def detokenize(self, sentence: list[int]) -> str:
        list_words = [self.index_to_word[index] for index in sentence]
        return ' '.join(list_words)

    def save_tokenizer(self, name: str):
        if not os.path.exists(self.PATH_DATASET):
            os.makedirs(self.PATH_DATASET)
        with open(self.PATH_DATASET + '/' + name, 'wb') as f:
            pickle.dump(self, f)

    def load_tokenizer(self, name: str):
        with open(self.PATH_DATASET + '/' + name, 'rb') as f:
            x = pickle.load(f)
            for key, value in x.__dict__.items():
                setattr(self, key, value)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, CustomTokenizer):
            return NotImplemented
        else:
            prop_names1 = list(self.__dict__)
            prop_names2 = list(__o.__dict__)
            n = len(prop_names1)
            for i in range(n):
                if not getattr(self, prop_names1[i]).__eq__(getattr(__o, prop_names2[i])):
                    return False
            return True
