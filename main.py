import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer=PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tok_sentence,words):
    tok_sentence=[stem(w) for w in tok_sentence]
    bag=np.zeros(len(words),dtype=np.float32)

    for indx,w in enumerate(words):
        if w in tok_sentence:
            bag[indx]=1.0
    return bag 

    
# a="Synchrony Financial is a consumer financial services company headquartered in Stamford, Connecticut, United States"
# a=tokenize(a)
# print(a)
# c=['babies','baby','babys']
# result = [stem(w) for w in c]
# print(result)
#pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
