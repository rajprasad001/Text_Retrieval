import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words('english')

class Data_preprocessing:
    def __init__(self, data):
        self.data = data

    def lower_case(self):
        data = self.data
        temp_list = []
        for content in data['text']:
            temp = content.lower()
            temp_list.append(temp)
        data['processed_text'] = temp_list
        return data

    def punctuation_removal(self):
        data = self.data
        temp_list = []
        for content in data['processed_text']:
            for c in string.punctuation:
                content = content.replace(c, "")
            temp_list.append(content)
        data = data.drop(columns='processed_text', axis=1)
        data['processed_text'] = temp_list
        return data

    def stopword_removal(self):
        data = self.data
        temp_list = []
        for content in data['processed_text']:
            word_tokens = word_tokenize(content)
            clean_text = [word for word in word_tokens if not word in stop_words]
            temp_list.append(clean_text)
        data = data.drop(columns='processed_text', axis=1)
        data['processed_text'] = temp_list

        return data


