#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Thu Jan 18 14:16:26 2018

Preparing text data for seq2seq chatbot

"""

import re
import unicodedata
import json
from collections import Counter
import os

import sys

BOS_TOKEN_id = 0   #beginning of sentence token index
EOS_TOKEN_id = 1   #end of sentence token index
OOV_TOKEN_id = 2   #out of vocabulary token index

BOS_TOKEN = '<BOS>'  #beginning of sentence token
EOS_TOKEN = '<EOS>'  #end of sentence token
OOV_TOKEN = 'UNK'    #out of vocabulary token
EOC_TOKEN = '\n'  #end of conversation token

MIN_OCCURENCE = 3
MAX_LENGTH = 30

def unicodeToAscii(s):
    # Convert unicode to ASCII
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = s.replace('...', '#')               #count ... as a distinct punctuation
    s = re.sub(r"([.!?#])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?#]+", r" ", s)
    s = s.replace('#', '...')
    return s

def createLinesDic(file, split_token=' +++$+++ '):
    lines_dic = {}
    for line in file:
        # Normalizing strings and storing them in dict with key being line number
        temp = line.strip().split(split_token)
        lines_dic[temp[0]] = normalizeString(temp[-1])
    return lines_dic

def createConversationDataset(input_file, lines_file, output_file):
    #saving preprocessed conversation in distinct file

    lines_dic = createLinesDic(lines_file)

    for conv in input_file:
        temp = conv.strip().split(' +++$+++ ')[-1]
        temp = ''.join(i for i in temp if i.isalnum() or i.isspace()).split()
        for line in temp:
            output_file.write(lines_dic[line] + ' ' + EOS_TOKEN + ' ')
        output_file.write(EOC_TOKEN)

def createLookupDic(conversations):

    word2id = {BOS_TOKEN: BOS_TOKEN_id,
               EOS_TOKEN: EOS_TOKEN_id,
               OOV_TOKEN: OOV_TOKEN_id}
    word2count = {}
    id2word = {BOS_TOKEN_id: BOS_TOKEN,
               EOS_TOKEN_id: EOS_TOKEN,
               OOV_TOKEN_id: OOV_TOKEN}

    vocab_size = len(id2word)

    # Count word occurences
    for conv in conversations:
        for word in conv.strip().split():
            if word in word2count:
                word2count[word] += 1
            else:
                word2count[word] = 1

    # Assign word to ix and vice versa
    # Word mapped to UNK if occurence < min_occurence
    for word in word2count:
        if word2count[word] >= MIN_OCCURENCE:
            if word in word2id:     #make sure there is no overwrite (BOS, EOS, ..)
                continue
            word2id[word] = vocab_size
            id2word[vocab_size] = word
            vocab_size += 1

    # Save word to index lookup table
    with open('word2id.json', 'w') as outfile:
        json.dump(word2id, outfile)

    # Save index to word lookup table
    with open('id2word.json', 'w') as outfile:
        json.dump(id2word, outfile)

def convertWordSentenceToIndex(sentence, word2id, online=False):

    if online:
        # additional preprocessing
        sentence = normalizeString(sentence)
    new_sentence = []

    for word in sentence.split():
        if word in word2id:
            new_sentence.append(word2id[word])
        else:
            new_sentence.append(OOV_TOKEN_id)
    return new_sentence

def convertIndexSentenceToWord(sentence, id2word):

    new_sentence = ''.join((id2word[str(i)] + ' ') for i in sentence)

    return new_sentence

def createPairs(conversations, word2id, id2word):
    pairs = []
    for conv in conversations:
        # convert the conversation to indexes
        conv = convertWordSentenceToIndex(conv, word2id)

        #save in pairs array as input and target entries
        sentence = []
        sentences = []              #keep track of all sentences in conversation
        for word in conv:
            sentence.append(word)
            if word == EOS_TOKEN_id:
                #encountered end of sentence
                sentences.append(sentence)
                sentence = []

        for i in range(len(sentences) - 1):
            if len(sentences[i]) <= MAX_LENGTH and \
                len(sentences[i+1]) <= MAX_LENGTH:
                #we discart both pairs to which this sentence belongs
                pairs.append({})
                pairs[-1]['input'] = sentences[i]
                pairs[-1]['target'] = sentences[i + 1]

    return pairs

def loadDataset(dataset_name='data/conversations.txt', generate_new=False):

    print("Importing and preprocessing data ...")

    if not os.path.exists('data/conversations.txt') or generate_new:
        # If not already done, create conversations.txt, where the dialogs are
        #arranged in order
        movie_lines = open('data/cornell_movie_dialogs_corpus/movie_lines.txt',
                           encoding='ISO-8859-2')
        movie_conversations = open('data/cornell_movie_dialogs_corpus/movie_conversations.txt')

        dataset = open(dataset_name, 'w')

        createConversationDataset(movie_conversations,
                                  movie_lines, dataset)

        dataset.close()

    if not os.path.exists('word2id.json') or generate_new:
        # If the word 2 index (and vice versa) lookup tables haven't been created yet
        dataset = open(dataset_name)
        createLookupDic(dataset)
        dataset.close()

    dataset = open(dataset_name)

    word2id = json.load(open('word2id.json'))
    id2word = json.load(open('id2word.json'))

    pairs = createPairs(dataset, word2id, id2word)

    vocab_size = len(word2id)

    return pairs, vocab_size, word2id, id2word


#--------------------------------tests-----------------------------------------

if __name__ == "__main__":

    pairs, vocab_size, word2id, id2word = loadDataset()
   
