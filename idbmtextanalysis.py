
""" 

"""

import xlsxwriter
from argparse import ArgumentParser
import sys
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

class IdbmTextClassifier():
    
    def __init__(self, file1, file2):
        
        self.train_data = pd.read_csv(file1)
        self.test_data = pd.read_csv(file2)
        self.vectorizer = CountVectorizer(ngram_range=(1, 3))
        
    def training_data(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.train_data['text'].values, self.train_data['label'].values, random_state = 5)
        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)
        
        self.logreg = LogisticRegression(C=10, penalty = 'l2', solver = 'liblinear').fit(X_train, y_train )
        
        self.logtrain_score = self.logreg.score(X_train, y_train)
        self.logtest_score = self.logreg.score(X_test, y_test)
        
        user_input = input("Review the movie: ")
        
        print(self.logreg.predict(self.vectorizer.transform([user_input]))[0])
        
        
if __name__ == "__main__":
    parser = ArgumentParser(description='Insert training file and test file.')
    parser.add_argument('file1', type=str,
                    help='a file for the training files')
    parser.add_argument('file2', type=str,
                    help='a file for the test files')
    args = parser.parse_args(sys.argv[1:])
    
    file_use = IdbmTextClassifier(args.file1, args.file2)
    file_use.training_data()
    
    if file_use.training_data == 0:
        print("We predict that this review is negative")
    else:
        print("we predict that this review is postive")
    
        
