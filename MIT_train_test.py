import io
import re
import string
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stop_words = set(stopwords.words('english')) 

file_loc = "Labels.xlsx"
df = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols = "A,B")
sub_name = df["course"].tolist()


subject_name = []
final_txt = []
complete_path = []

base = "C:\\Users\\vnitin\\Documents\\Dataset\\"
# base = "C:\\Users\\vnitin\\Documents\\Processed\\"
for path in os.listdir(base):
    for file in os.listdir(os.path.join(base, path)):
        if file.endswith(".txt"):
            full_path = os.path.join((os.path.join(base, path)),file)
            complete_path.append(full_path)

            pdf = open(full_path, encoding="utf8")
            pdf_content = pdf.read().lower()
            cleared = re.sub(r'\b\w{1,3}\b', '', pdf_content)
            cleared_text = re.sub('[^a-zA-Z]+', ' ', cleared)
            
            processed_txt = ""
            subject_name.append(path)
        
            words = cleared_text.split()
            lemmatizer = WordNetLemmatizer()
            for r in words:
                if not r in stop_words:
                    processed_txt+=str(str(lemmatizer.lemmatize(r) + " "))
            final_txt.append(processed_txt)

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(final_txt, subject_name, complete_path,test_size = 0.2, random_state=42)
#Y_train, y_test = train_test_split(subject_name, test_size = 0.2,shuffle=False)
clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

print(len(X_train))
print(len(X_test))
print(len(z_train))
print(len(z_test))

clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

for doc, category in zip(z_test, predicted):
    print(doc,"=>",category)

print(np.mean(predicted == y_test))


"""
# tfidf_vect = TfidfVectorizer()
# tf_fit = tfidf_vect.fit_transform(X_train)
# clf = MultinomialNB().fit(tf_fit, y_train)


# docs_new = ['Electromagnetism is being generate from magnet and electricty','Register stores different value from the counter']
# X_new_tfidf = tfidf_vect.transform(docs_new)
# predicted = clf.predict(X_new_tfidf)

# X_test_tfidf = tfidf_vect.transform(X_test)
# test_predicted = clf.predict(X_test_tfidf)
"""
