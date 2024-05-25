import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.god_terms = ['god', 'jesus', 'allah', 'yahweh', 'buddha', 'shiva', 'krishna']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X['processed_text'] = X['text'].apply(self.process_text)
        X['processed_text_str'] = X['processed_text'].apply(lambda tokens: ' '.join(tokens))
        X['mention_god_related'] = X['processed_text'].apply(lambda x: 1 if any(term in x for term in self.god_terms) else 0)

        return X

    def process_text(self, text):
        cleaned = clean_text(text)
        lemmatized = lemmatize_sentence(cleaned)
        filtered = filter_stop_words(lemmatized)
        return filtered

    def set_output(self, *args, **kwargs):
        return self



def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text) # remove mentions
    text = re.sub(r'\[.*?\]', '', text) # remove [] and things inside it 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # remove links 
    text = re.sub(r'<.*?>+', '', text) # remove HTML tags
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = re.sub('\n', '', text) # remove newline
    text = re.sub(r'\w*\d\w*', '', text) # remove digits
    
    return text

# tagging words
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return 'a'  # Adjective
    elif nltk_tag.startswith('V'):
        return 'v'  # Verb
    elif nltk_tag.startswith('N'):
        return 'n'  # Noun
    elif nltk_tag.startswith('R'):
        return 'r'  # Adverb
    else:          
        return None

# lemmatization with POS tagging 
def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_sentence

# removing stop words
def filter_stop_words(tokenized_tweet):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokenized_tweet if word not in stop_words]

# processing text 
def processing_text(text): 
    text = clean_text(text)
    lematized = lemmatize_sentence(text)
    without_stopwords = filter_stop_words(lematized)
    return without_stopwords