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
from nltk.stem.snowball import SnowballStemmer,PorterStemmer
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
        # do not exceed 29 mins
        X = self.clean_all_data(X)
        
        # self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2',
        #                                     use_idf=True, smooth_idf=True, ngram_range=(1,5))
        # XX = self.preprocessor.fit_transform(X["combined_text"])
        # required_cat_features = ['state']

        required_text_features = ['combined_text']

        # required_binary_features = ['has_company_logo']
        # binary_transformer = Pipeline(steps=[('label', OneHotEncoder(handle_unknown='ignore'))])
        # cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])

        # text_transformer = Pipeline(steps=[('nltk', nltk.tokenize)])
        # text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                # ('bin', binary_transformer, required_binary_features),
                # ('cat', cat_transformer, required_cat_features),
                *[(feature_name, text_transformer, feature_name)
                  for feature_name in required_text_features]
            ]
        )

        # log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
        #                                ('classifier', SGDClassifier(class_weight="balanced"))])

        log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', LinearSVC())])

        self.clf = log_reg_pipe

        self.clf.fit(X, y)

        # print(XX)

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X = self.clean_all_data(X)

        required_text_features = ['combined_text']

        text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])

        # text_transformer = Pipeline(steps=[('nltk', nltk.tokenize)])
        # text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                # ('bin', binary_transformer, required_binary_features),
                # ('cat', cat_transformer, required_cat_features),
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

        numbers = ["0", "1","2","3","4","5","6","7","8","9"]

        df = pd.DataFrame(X)

        df['FEATURE REQUEST'] = df['FEATURE REQUEST'].str.lower()
        df['SMELLS'] = df['SMELLS'].str.lower()

        for char in spec_chars:
            df['FEATURE REQUEST'] = df['FEATURE REQUEST'].str.replace(char, ' ')
            df['SMELLS'] = df['SMELLS'].str.replace(char, ' ')

        for number in numbers:
            df['FEATURE REQUEST'] = df['FEATURE REQUEST'].str.replace(number, ' ')
            df['SMELLS'] = df['SMELLS'].str.replace(number, ' ')

        df['FEATURE REQUEST'] = df['FEATURE REQUEST'].str.replace('https?://\S+|www\.\S+', ' ')
        df['SMELLS'] = df['SMELLS'].str.replace('https?://\S+|www\.\S+', ' ')

        df['combined_text'] = df['FEATURE REQUEST'] + " " + df['SMELLS']
        # df['combined_text'] = df['FEATURE REQUEST']
        drop_cols = ['FEATURE REQUEST', 'SMELLS']
        df = df.drop(drop_cols, axis=1)
        
        self.all_genism_stop_words = STOPWORDS
        
        text_columns = list(df.columns.values)
        
        for columns in text_columns:
            self.remove_stopwords_from_data_train(df,columns)
        
        # Add lemmatization
        
        # Add tokenization
        # df = nltk.word_tokenize(df['combined_text'])
        df['combined_text'] = df['combined_text'].str.lower()
        df['tokens'] = df.apply(self.tokenization_df, axis = 1)
        df['stems'] = df.apply(self.stem_df,axis = 1)
        return df

    # def tokenization_df(self,data_frame):
    #     data_frame_token = []
    #     for index in data_frame.index:
    #         token = sent_tokenize(index)
    #         if token != '':
    #             data_frame_token.append(token)
    #     return data_frame_token

    def tokenization_df(self,data_frame):
        data_frame_token = data_frame['combined_text']
        tokens = nltk.sent_tokenize(data_frame_token)

        token_words = [words for words in tokens if words.isalpha()]
        return token_words

    def stem_df(self,data_frame):
        stemming = SnowballStemmer(language='english')
        data_frame_token = data_frame['combined_text']
        stemmed = [stemming.stem(word) for word in data_frame_token]
        return stemmed
    
    def remove_stopwords_from_data_train(self,data_frame, column_name):
        data_frame[column_name] = data_frame[column_name].apply(lambda x: " ".join([i for i in x.lower().split() if i not in self.all_genism_stop_words]))


