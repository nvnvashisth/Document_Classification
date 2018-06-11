import io
import re
import string
import os
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import goslate
from nltk.tokenize import RegexpTokenizer
#from gensim import corpora, models
#import gensim
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
#List of stopwords in English category
stop_words = set(stopwords.words('english')) 
#legal_characters = '^\w+$'
#translator = Translator()
#gs = goslate.Goslate()
file_loc = "Labels.xlsx"
df = pd.read_excel(file_loc, index_col=None, na_values=['NA'], usecols = "A,B")
sub_name = df["course"].tolist()
#print(sub_name)

subject_name = []
final_txt = []
#words = set(nltk.corpus.words.words())
base = "C:\\Users\\vnitin\\Documents\\Dataset\\"
for path in os.listdir(base):
    for file in os.listdir(os.path.join(base, path)):
        if file.endswith(".txt"):
            full_path = os.path.join((os.path.join(base, path)),file)
        #PDF read content from text file
            pdf = open(full_path, encoding="utf8")
            pdf_content = pdf.read().lower()
            cleared = re.sub(r'\b\w{1,3}\b', '', pdf_content)
            cleared_text = re.sub('[^a-zA-Z]+', ' ', cleared)
            
            #meaning_words = " ".join(w for w in nltk.wordpunct_tokenize(cleared_text) if w.lower() in words or not w.isalpha())
            processed_txt = ""
            frequency = {}
            subject_name.append(path)
        
            words = cleared_text.split()
            lemmatizer = WordNetLemmatizer()
            for r in words:
                if not r in stop_words:
                    processed_txt+=str(str(lemmatizer.lemmatize(r) + " "))
            final_txt.append(processed_txt)
#print(final_txt)
#print(subject_name)
# count_vect = CountVectorizer()
tfidf_vect = TfidfVectorizer()

# count_fit = count_vect.fit_transform(final_txt)
tf_fit = tfidf_vect.fit_transform(final_txt)
#print(tf_fit)

clf = MultinomialNB().fit(tf_fit, subject_name)
#print(clf)

docs_new = ['Electromagnetism is being generate from magnet and electricty','Register stores different value from the counter']
#X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_vect.transform(docs_new)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print(doc,"=>",category)