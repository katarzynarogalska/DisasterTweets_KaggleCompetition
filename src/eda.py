import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from textblob import TextBlob
from nrclex import NRCLex #detect more emotions
from sklearn.base import BaseEstimator, TransformerMixin


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique=set()
    def fit(self, X, y=None):
        for emotion_list in X['text'].apply(find_emotions):
            self.unique.update(emotion_list)
        return self

    def transform(self, X):
       
        X['count_caps_lock'] = X['text'].apply(count_caps_lock)
        X['count_exclamation_mark'] = X['text'].apply(count_exclamation_mark)
        X['count_hashtags'] = X['text'].apply(count_hashtags)
        X['count_words'] = X['text'].apply(count_words)
        X['count_punctuation'] = X['text'].apply(count_punctuation)
        X['count_links'] = X['text'].apply(count_links)
        X['count_stopwords'] = X['text'].apply(count_stopwords)
        X['count_mentions'] = X['text'].apply(count_mentions)
        X['count_verbs'] = X['text'].apply(count_parts_of_speech, tag='V')
        X['count_nouns'] = X['text'].apply(count_parts_of_speech,tag= 'N')
        X['count_adjectives'] = X['text'].apply(count_parts_of_speech, tag='J')
        X['count_adverbs'] = X['text'].apply(count_parts_of_speech,tag= 'RB')
        X['polarity'] = X['text'].apply(get_polarity)
        X['subjectivity'] = X['text'].apply(get_subjectivity)
        X['emotions'] = X['text'].apply(find_emotions)


        for e in self.unique:
            X[e] = X['emotions'].apply(lambda x: 1 if e in x else 0)

        return X

    def set_output(self, *args, **kwargs):
        return self


def count_caps_lock(text):
    words = text.split()
    caps_lock_words = [word for word in words if word.isupper()]
    return len(caps_lock_words)

def count_exclamation_mark(text):
    return len(re.findall(r'!', text))

def count_hashtags(text):
    return len(re.findall(r'#\w+', text))

def count_words(text):
    return len(text.split())

def count_punctuation(text):
    punctuation_pattern = r'[^\w\s]'
    punct = re.findall(punctuation_pattern, text)
    return len(punct)

def count_links(text):
    url_pattern = r'https?://\S+|www\.\S+'
    links=re.findall(url_pattern,text)
    return len(links)

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    sw = [word for word in text.split() if word in stop_words]
    return len(sw)

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    sw = [word for word in text.split() if word in stop_words]
    return len(sw)

def count_mentions(text):
    mention_pattern = r'@\w+'
    mentions=re.findall(mention_pattern,text)
    return len(mentions)

def count_parts_of_speech(text, tag):
    tokens= word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    part_count = sum(1 for word,pos in tagged if pos.startswith(tag))
    return part_count
    
def get_polarity(text): # checking if a tweet was negative (-1), neutral (0) or positive (1)
    return TextBlob(text).sentiment.polarity

def get_subjectivity(text): #checking if a tweet is subjective or objective
    return TextBlob(text).sentiment.subjectivity

def find_emotions(text):
    emotion = NRCLex(text)
    top = emotion.top_emotions
    emotion_names = [emotion_tuple[0] for emotion_tuple in top]
    return list(emotion_names)




    