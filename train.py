# encoding: utf-8
# 2019/03/15 16:28: TfidfVectorizer save
#

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from janome.tokenizer import Tokenizer
import pickle

#
def get_token(text):
    t = Tokenizer()
    tokens = t.tokenize(text, wakati=True)  # 分かち書きする
    word = ""
    for token in tokens:
        word +=token  + " "
    return word
#
words1="利用人数は何人ですか？"
words2="契約期間は、ありますか？"
words3="オープンソースですか？"
words4="オンライン決済は、可能ですか?"
words5="製品価格、値段はいくらですか？"

#words= get_token(words1 )
#print(words )
#quit()
words =[]
words.append(words1 )
words.append(words2 )
words.append(words3 )
words.append(words4 )
words.append(words5 )

#print(words )
tokens=[]
for item in words:
    token=get_token(item)
    tokens.append(token)
#
#print(tokens )
#quit()
docs = np.array(tokens)

vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
print(tokens)
#quit()
vecs = vectorizer.fit_transform(docs )
print("#vecs :")
print(vecs.shape )
##print(vecs[0] )

#save
file_name="params.pkl"
with open(file_name, 'wb') as f:
    pickle.dump(vectorizer, f)
print("#save vectorizer OK!")
