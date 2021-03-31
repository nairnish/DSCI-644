import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from gensim.parsing.preprocessing import STOPWORDS
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
        # drop_cols = ['telecommuting', 'has_questions', 'country', 'city']
        # X = X.drop(drop_cols, axis=1)
        # bin_features = ['telecommuting', 'has_company_logo', 'has_questions']
        # text_features = ['title', 'description', 'requirements']
        # cat_features = ['country', 'state', 'city']
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
        #                                ('classifier', LogisticRegression())])
        # self.clf = log_reg_pipe
        predictions = self.clf.predict(X)
        # probs = self.clf.predict_proba(X)
        return predictions

        
    def clean_all_data(self, X):
        #warnings.filterwarnings(action='ignore')

        #fillna to location column
        # data_frame['location'] = data_frame.location.fillna('none')
        #
        # #fillna to description column
        # data_frame['description'] = data_frame.description.fillna('not specified')
        #
        # #fillna to requirements column
        # data_frame['requirements'] = data_frame.description.fillna('not specified')
        #
        # #drop unnecassary columns
        # data_frame.drop(['telecommuting','has_questions'],axis = 1, inplace = True)
        #
        # #mapping fraudulent to T and F, where there is  0 and 1 respectively
        # data_frame['has_company_logo'] = data_frame.has_company_logo.map({1 : 't', 0 : 'f'})
        
        # #remove any unnecassary web tags in the data set
        # data_frame['title'] = data_frame.title.str.replace(r'<[^>]*>', '')
        # data_frame['description'] = data_frame.description.str.replace(r'<[^>]*>', '')
        # data_frame['requirements'] = data_frame.requirements.str.replace(r'<[^>]*>', '')


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
        
        
        # removing the characters in data set that are not words and has white spaces
        # for column in data_frame.columns:
        #     data_frame[column] = data_frame[column].str.replace(r'\W', ' ').str.replace(r'\s$','')
            
        
        # mapping back the columns to original binary values
        #data_frame['has_company_logo'] = data_frame.has_company_logo.map({'t': 1, 'f':0})


        
        self.all_genism_stop_words = STOPWORDS
        
        text_columns = list(df.columns.values)
        
        for columns in text_columns:
            self.remove_stopwords_from_data_train(df,columns)
        
        return df
    
    def remove_stopwords_from_data_train(self,data_frame, column_name):
        data_frame[column_name] = data_frame[column_name].apply(lambda x: " ".join([i for i in x.lower().split() if i not in self.all_genism_stop_words]))
