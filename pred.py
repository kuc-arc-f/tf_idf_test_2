# encoding: utf-8
# 2019/03/15 16:28: transform add.
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
docs = np.array(tokens)
#
file_name="params.pkl"
vectorizer =None
with open(file_name, 'rb') as f:
    vectorizer = pickle.load(f)
print("load vectorizer OK!!")

#
vecs= vectorizer.transform( docs )

#print(type(vectorizer ))
#quit()
#print(vecs.shape )
#print(vecs )
#quit()
#print(tokens)
#str="利用人数は？"
#str="契約期間"
str="価格は？"

instr = get_token(str ).strip()
print("instr=", instr )
x= vectorizer.transform( [  instr ])

#print( "x=",x)
#Cosine類似度（cosine_similarity）の算出
num_sim=cosine_similarity(x , vecs)
print(num_sim )
index = np.argmax( num_sim )

print("word=", words[index])
print()
    