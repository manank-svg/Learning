import nltk as nlp

# nlp.download("all")
# Tokenization using NLTK
from nltk import word_tokenize, sent_tokenize
sent = "W3Schools is a great learning platform.\
It is one of the best for Computer Science students."
print(word_tokenize(sent))
print(sent_tokenize(sent))
# Stemming using NLTK
from nltk.stem import PorterStemmer
# initial
# create an object of class PorterStemmer
porter = PorterStemmer()
print(porter.stem("running"))
print(porter.stem("playing"))
print(porter.stem("Gymnastics"))
print(porter.stem("Dancing"))

# Lemmatization using NLTK
from nltk.stem import WordNetLemmatizer
# create an object of class WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("Better", 'v'))
print(lemmatizer.lemmatize("played", 'v'))
print(lemmatizer.lemmatize("play", 'v'))
print(lemmatizer.lemmatize("playing", 'v'))

# Part-of-Speech (POS) Tagging using NLTK
from nltk import pos_tag

text = "GeeksforGeeks is a Computer Science platform."
tokenized_text = word_tokenize(text)
tags = pos_tag(tokenized_text)
print(tags)