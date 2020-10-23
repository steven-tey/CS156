import numpy as np
from collections import Counter
import pickle

def getWordList(data):
    '''
    This function takes in the response data and converts it
    into one huge string, and then uses a Counter to identify 
    unique words and the number of occurences
    '''
    conversations = open(data, 'r')
    allLines = conversations.readlines()
    combinedLines = ""
    for line in allLines:
        combinedLines += line
    wordDict = Counter(combinedLines.split())
    return combinedLines, wordDict

# run the getWordList function on the responseData.txt file
fullCorpus, convoDict = getWordList('responseData.txt')
print('Converted response data into full corpus and conversation dictionary.')

# get all the keys in the dictionary 
# (the values are the number of occurences of each key)
wordList = list(convoDict.keys())

# Pickle dump the wordlist into .txt file
with open("wordList.txt", "wb") as fp:
    pickle.dump(wordList, fp)