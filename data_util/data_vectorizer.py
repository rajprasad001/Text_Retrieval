import string
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


class Data_Preprocessing:
    def __init__(self, data):
        self.data = data

    def punctuation_removal(self):
        data = self.data
        temp_list = []
        for content in data['text']:
            for c in string.punctuation:
                content = content.replace(c, " ")
            temp_list.append(content)
        data['processed_text'] = temp_list
        print(' 1. Punctuation Removal successful.')
        return data

    def lemmatizing(self):
        data = self.data
        temp_sent = []
        for content in data['processed_text']:
            word_list = nltk.word_tokenize(content)
            lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
            temp_sent.append(lemmatized_output)
        data = data.drop(columns='processed_text', axis=1)
        data['processed_text'] = temp_sent
        print(' 2. Lemmatization Successful')
        return data


class Vectorization:
    def __init__(self, data):
        self.preprocessed_data = data

    def vec_gen(self):
        content_list = []
        dataframe = self.preprocessed_data

        for content in dataframe['processed_text']:
            content_list.append(content)

        tf_idf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        doc_term_matrix = tf_idf_vectorizer.fit_transform(content_list)
        return tf_idf_vectorizer, doc_term_matrix


class User_Input_Vectorizer:
    def __init__(self, data, tf_idf_vectorizer):
        self.user_input = data
        self.tfidf_vectorizer = tf_idf_vectorizer

    def input_preprocessing(self):
        tfidf_vectorizer = self.tfidf_vectorizer
        input_text = self.user_input
        for punc in string.punctuation:
            input_text = input_text.replace(punc, ' ')
        tokenized_text = nltk.word_tokenize(input_text)
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokenized_text])
        input_term_matrix = tfidf_vectorizer.transform([lemmatized_output])
        input_embedding = pd.DataFrame(input_term_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
        return input_embedding
