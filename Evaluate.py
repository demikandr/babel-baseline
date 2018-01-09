#/bin/python
# coding: utf-8

# In[2]:

from __future__ import unicode_literals, print_function, division
from io import open
import string
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils import Lang, unicodeToAscii, normalizeString,                      indexesFromSentence, variableFromSentence, use_cuda,                      SOS_token, EOS_token
from src.model import EncoderRNN, AttnDecoderRNN, MAX_LENGTH
# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')


# In[3]:

use_cuda


# In[4]:

MAX_LENGTH = 100

def filterSample(p):
    return True
    return len(p.split(' ')) < MAX_LENGTH


def filterSamples(samples):
    return [' '.join(sample.split(' ')[:MAX_LENGTH - 2]) for sample in samples if filterSample(sample)]


# In[5]:

import pickle
def readLang(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    filename = '../../../temp/Batch2a_%s.txt' % (lang1)
    filename = '/data/input.txt'
    lines_lang = open(filename, encoding='utf-8').        read().strip().split('\n')

    # Split every line into pairs and normalize
    samples = [normalizeString(s) for s in lines_lang]

    # Reverse pairs, make Lang instances
    input_lang = pickle.load(open(lang1 + '.lang', 'rb'))
    output_lang = pickle.load(open(lang2 + '.lang', 'rb'))

    return input_lang, output_lang, samples
def prepareData(lang1, lang2):
    input_lang, output_lang, samples = readLang(lang1, lang2)
    print("Read %s sentence samples" % len(samples))
    samples = filterSamples(samples)
    print("Trimmed to %s sentence samples" % len(samples))
    return input_lang, output_lang, samples


input_lang, output_lang, samples = prepareData('de', 'en')
print(random.choice(samples))


# In[42]:

from numpy.random import choice
import numpy as np

def softmax(x):
    x = x - x.max()
    return np.exp(x) / np.sum(np.exp(x), keepdims=True)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
#         topv, topi = decoder_output.data.topk(1)
#         print(decoder_output.data[0].shape)
        ni = choice(np.arange(0, len(decoder_output.data[0])), p=softmax(decoder_output.data[0].numpy()))
        ni = int(ni)
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


# In[43]:

from tqdm import tqdm
def evaluateIters(encoder, decoder):
    translations = []
    for iter in tqdm(range(len(samples))):
        evaluated_sentence = samples[iter]
        translation = evaluate(encoder, decoder, evaluated_sentence)
        translations.append(translation)
    return translations


# In[44]:

def load_params(encoder, decoder):
    encoder.load_state_dict(torch.load('./encoder.params'))
    decoder.load_state_dict(torch.load('./decoder.params'))


# In[45]:

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               1, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

load_params(encoder1, attn_decoder1)
translations = evaluateIters(encoder1, attn_decoder1)
translations = [' '.join(translation[0][:-1]) for translation in translations]
with open('/output/output.txt', 'w') as file:
    file.write('\n'.join(translations))


# In[ ]:

# for sample, translation in zip(samples, translations):
#     print(sample, '\t#\t', translation)
#     input()

