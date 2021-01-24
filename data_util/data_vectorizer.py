import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')


stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


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
        print(' 1. Lower casing successful.')
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
        print(' 2. Punctuation Removal successful.')
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
        print(' 3. Stopwords removal successful')
        return data

    def lemmatizing(self):
        data = self.data
        temp_list = []
        temp_sent = []
        for content in data['processed_text']:
            for word in content:
                lem_word = lemmatizer.lemmatize(word, pos='v')
                #print("{0:20}{1:20}".format(word,lem_word))
                temp_sent.append(lem_word)
            temp_list.append(temp_sent)
            temp_sent = []
        data = data.drop(columns='processed_text', axis=1)
        data['processed_text'] = temp_list
        print(' 4. Lemmatization Successful')
        return data


class Vectorization:
    def __init__(self, data):
        self.preprocessed_data = data

    def vec_gen(self):
        dataframe = self.preprocessed_data
        tfidf_vectorizer = TfidfVectorizer()
        fit_vect = tfidf_vectorizer.fit_transform(dataframe.processd_text.values)
        print (fit_vect)
        return fit_vect



