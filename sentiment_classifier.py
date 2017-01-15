import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names


class SentimentClassifier():
    def __init__(self):
        positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)']
        negative_vocab = ['bad', 'terrible', 'useless', 'hate', ':(']
        neutral_vocab = ['movie', 'the', 'sound', 'was', 'is', 'actors', 'did', 'know', 'words', 'not']
    
        positive_features = [(self.word_feats(pos), 'pos') for pos in positive_vocab]    
        negative_features = [(self.word_feats(neg), 'neg') for neg in neutral_vocab]
        neutral_features = [(self.word_feats(neu), 'neu') for neu in neutral_vocab]
    
        train_set = negative_features + positive_features + neutral_features
    
        self.classifier = NaiveBayesClassifier.train(train_set)
    
    def word_feats(self, words):
        return dict([(word, True) for word in words])    
    
