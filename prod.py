import warnings

import nltk as nltk
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC, LinearSVC
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re
from nltk import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.parsing.preprocessing import remove_stopwords
import nltk


class my_model():

    def fit(self, X, y):
        X = self.clean_all_data(X)

        required_text_features = ['combined_text']

        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])

        preprocessor = ColumnTransformer(
            transformers=[
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )

        log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', LinearSVC())])

        self.clf = log_reg_pipe

        self.clf.fit(X, y)

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X = self.clean_all_data(X)
        
        required_text_features = ['combined_text']

        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])

        preprocessor = ColumnTransformer(
            transformers=[
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )

        predictions = self.clf.predict(X)
        # probs = self.clf.predict_proba(X)
        return predictions

    def clean_all_data(self, X):
        warnings.filterwarnings(action='ignore')

        spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                      "*", "+", ",", "-", ".", "/", ":", ";", "<",
                      "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                      "`", "{", "|", "}", "~", "–", "\\n"]

        numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        df = pd.DataFrame(X)

        # Converting all text to lower case
        df['FEATUREREQUEST'] = df['FEATUREREQUEST'].str.lower()
        df['SMELLS'] = df['SMELLS'].str.lower()

        # Removal of special characters
        for char in spec_chars:
            df['FEATUREREQUEST'] = df['FEATUREREQUEST'].str.replace(char, ' ')
            df['SMELLS'] = df['SMELLS'].str.replace(char, ' ')

        # Removal of numbers
        for number in numbers:
            df['FEATUREREQUEST'] = df['FEATUREREQUEST'].str.replace(number, ' ')
            df['SMELLS'] = df['SMELLS'].str.replace(number, ' ')

        # Removal of web tags
        df['FEATUREREQUEST'] = df['FEATUREREQUEST'].str.replace('https?://\S+|www\.\S+', ' ')
        df['SMELLS'] = df['SMELLS'].str.replace('https?://\S+|www\.\S+', ' ')

        # Combining text features
        df['combined_text'] = df['FEATUREREQUEST'] + " " + df['SMELLS']

        # dropping columns that are not required
        drop_cols = ['FEATUREREQUEST', 'SMELLS']
        df = df.drop(drop_cols, axis=1)

        # Stopword removal
        self.all_genism_stop_words = STOPWORDS
        text_columns = list(df.columns.values)
        for columns in text_columns:
            self.remove_stopwords_from_data_train(df, columns)
        
        # Stemming
        df.apply(self.stem_df, axis=1)
        # Lemmatization
        df.apply(self.lemma_df, axis=1)

        df['stems'] = df.apply(self.stem_df, axis=1)
        df['lemma'] = df.apply(self.lemma_df, axis=1)
        return df

    def stem_df(self, data_frame):
        stemming = SnowballStemmer(language='english')
        data_frame_token = data_frame['combined_text']
        stemmed = [stemming.stem(word) for word in data_frame_token]
        return stemmed

    def lemma_df(self, data_frame):
        lemming = WordNetLemmatizer()
        data_frame_lemma = data_frame['combined_text']
        lemma = [lemming.lemmatize(word) for word in data_frame_lemma]
        return lemma

    def remove_stopwords_from_data_train(self, data_frame, column_name):
        data_frame[column_name] = data_frame[column_name].apply(
            lambda x: " ".join([i for i in x.lower().split() if i not in self.all_genism_stop_words]))


