import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
from sentiment_classifier import SentimentClassifier


def main():
    print("Enter the text to be analysed")
    sentence = raw_input()
    sentence = sentence.lower()
    words = sentence.split(' ')
    positive_cnt = 0
    negative_cnt = 0
    sentiment_classifier = SentimentClassifier()
    for word in words:
        result = sentiment_classifier.classifier.classify(sentiment_classifier.word_feats(word))
        if result == 'neg':
            negative_cnt = negative_cnt + 1
        if result == 'pos':
            positive_cnt = positive_cnt + 1
    
    print('Positive: ' + str(float(positive_cnt)/len(words)))
    print('Negative: ' + str(float(negative_cnt)/len(words)))        

if __name__ == "__main__":
    main()