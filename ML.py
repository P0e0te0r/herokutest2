import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')

df = pd.read_excel(r"C:\Users\Test\Documents\Machine Learning\Datensätze\iris\Iris.xlsx")
df.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head()

from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# logistic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# model training
classifier.fit(x_train, y_train)

# print metric to get performance
print("Accuracy: ", classifier.score(x_test, y_test) * 100)



import pickle

with open(r'C:\Users\Test\Documents\Machine Learning\Web_Anwendung\model_pickle', 'wb') as f:
    pickle.dump(classifier, f)
    


