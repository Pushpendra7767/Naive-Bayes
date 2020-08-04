# import dataset
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB
import re
from sklearn.model_selection import train_test_split
# print dataset
email_data = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Naive Bayes\\smsrawNB.csv",encoding = "ISO-8859-1")
# iumport stop words
stop_words = []
with open("C:\\Users\\ACER\\Desktop\\scrap\\Naive Bayes\\stopwords.txt") as f:
    stop_words = f.read()
# split stop words
stop_words = stop_words.split("\n")

"this is awsome 1231312 $#%$# a i he yu nwj"
# function for cleaning data
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

"This is Awsome 1231312 $#%$# a i he yu nwj".split(" ")

cleaning_text("This is Awsome 1231312 $#%$# a i he yu nwj")
cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi how .. are you?")
email_data.text = email_data.text.apply(cleaning_text)
email_data.shape
email_data = email_data.loc[email_data.text != " ",:]
# function for split into words
def split_into_words(i):
    return [word for word in i.split(" ")]
# split dataset into train & test
email_train,email_test = train_test_split(email_data,test_size=0.3)
emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.text)

all_emails_matrix = emails_bow.transform(email_data.text)
all_emails_matrix.shape 

train_emails_matrix = emails_bow.transform(email_train.text)
train_emails_matrix.shape

test_emails_matrix = emails_bow.transform(email_test.text)
test_emails_matrix.shape
# apply gausion & multinomial NB and print test pred, accuracy values
classifier_mb = MB()
classifier_mb.fit(train_emails_matrix,email_train.type)
train_pred_m = classifier_mb.predict(train_emails_matrix)
accuracy_train_m = np.mean(train_pred_m==email_train.type)

test_pred_m = classifier_mb.predict(test_emails_matrix)
accuracy_test_m = np.mean(test_pred_m==email_test.type)

classifier_gb = GB()
classifier_gb.fit(train_emails_matrix.toarray(),email_train.type.values) 
train_pred_g = classifier_gb.predict(train_emails_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==email_train.type)

test_pred_g = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==email_test.type) 
























