from winreg import REG_FULL_RESOURCE_DESCRIPTOR
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

import tweepy

import tweepy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import torch
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import os

import time


TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
device = torch.device('cpu')

SEED = 1234

# PARAMETERS

LEARNING_RATE = 5e-5
EPSILON_VALUE = 1e-8 # Affects the speed of training progress when using adam optimizer

# MODEL PARAMS

INPUT_DIM = 768
HIDDEN_LAYER_1_DIM = 64
HIDDEN_LAYER_2_DIM = 8
# Using output size 2 to get both the positive and negative probabilities
OUTPUT_DIM = 2
DROPOUT = 0.2
import random

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

from transformers import BertTokenizer

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

bert = BertModel.from_pretrained('bert-base-uncased')

class BertClassifier(nn.Module):
    def __init__(self, D_in, H1, H2, D_out, dropout):
        super(BertClassifier, self).__init__()

        # Instantiate the BERT model
        self.bert = bert

        # Hidden layers
        self.fc1 = nn.Linear(D_in,H1)
        self.fc2 = nn.Linear(H1,H2)

        # Output layer
        self.output = nn.Linear(H2,D_out)

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = output[0][:, 0, :]

        output = self.fc1(last_hidden_state_cls)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.output(output)

        return output

def initialize_model(lr,eps, D_in, H1, H2, D_out, dropout):
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(D_in, H1, H2, D_out, dropout)
    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)
    return bert_classifier

def preprocess(text, stem=False):
    tokens = []
    for token in text.split():
        if token not in stop_words or token in ['not', 'can', 'cannot', "can't", "won't", "don't", "shouldn't", "couldn't", "musn't", "woudldn't", "doesn't", "didn't", "no" ]:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def predict_sentiment(model,sentence):
    sentence = re.sub(TEXT_CLEANING_RE, ' ', str(sentence).lower()).strip()
    model.eval()
    encoded_sent = tokenizer.encode_plus(
            text=sentence,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=64,             # Max length to truncate/pad
            truncation=True,                # Truncate if sent exceeds max len.
            padding = 'max_length',         # Pad sentence to max length
            return_attention_mask=True      # Return attention mask (to ignore - padded tokens)
    )
    input_ids = [encoded_sent.get('input_ids')]
    attn_mask = [encoded_sent.get('attention_mask')]
    input_ids = torch.tensor(input_ids)
    attn_mask = torch.tensor(attn_mask)
    prediction = model(input_ids.to(device), attn_mask.to(device))
    label = prediction[0]
    return label

model = initialize_model(LEARNING_RATE,EPSILON_VALUE,INPUT_DIM,HIDDEN_LAYER_1_DIM, HIDDEN_LAYER_2_DIM, OUTPUT_DIM, DROPOUT)

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import json

channel_layer = get_channel_layer()

model.load_state_dict(torch.load('tweet-classifier-model.pt', map_location=device))

def get_sentiment_data(text):
    sentiment = F.sigmoid(predict_sentiment(model,text)).tolist()
    if abs(sentiment[0]-sentiment[1])<0.025:
        return "NEUTRAL", 0
    else:
        if sentiment[0]>sentiment[1]:
            return "NEGATIVE", sentiment[0]
        else:
            return "POSITIVE", sentiment[1]

class TweetListener(tweepy.StreamingClient):

    def on_data(self, data):
        data = json.loads(data)
        if data['data']['geo']:
            print(data['data']['geo'])
        tweet = re.sub(TEXT_CLEANING_RE, ' ', str(data['data']['text']).lower()).strip()
        if(len(tweet)>15):
            sentiment, val_weight = get_sentiment_data(tweet)
            tags = nltk.tag.pos_tag(tweet.split())
            action_words = []
            for tag in tags:
                if tag[1] in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'MD', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    action_words.append(tag[0])
            word_map = {}
            for word in action_words:
                senti, val = get_sentiment_data(word)
                word_map[word] = {
                    'sentiment': senti,
                    'weight': val
                }
            async_to_sync(channel_layer.group_send)('senti', {'type': 'send_senti', 'data': {'sentiment':sentiment, 'weight': val_weight,'tweet': data['data']['text'], 'word_map': word_map, 'time': time.time()}})

    def on_connection_error(self):
        self.disconnect()

streaming_client = TweetListener("AAAAAAAAAAAAAAAAAAAAAOm7bAEAAAAAbruy3eqX1mffF8UlA2DLQ1cv0jM%3Djso83M3MyeJ3hyjKfRnafWgGwL4mxoxMclPZ1Pn5JMGM5AcWdf", wait_on_rate_limit=True)



def get_rules_ids():
    ids = []
    for rule in streaming_client.get_rules().data:
        print(rule)
        ids.append(rule.id)
    return ids

class TweetConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add('senti', self.channel_name)
        await self.accept()

    async def disconnect(self,code): 
        await self.channel_layer.group_discard('senti', self.channel_name)
        streaming_client.disconnect()

    async def receive(self, text_data):
        try:
            streaming_client.delete_rules(get_rules_ids())
            streaming_client.disconnect()
        except:
            pass
        streaming_client.add_rules(tweepy.StreamRule("({0}) lang:en -is:retweet".format(text_data)))
        
        streaming_client.filter(expansions="geo.place_id", tweet_fields="text", place_fields="country",threaded=True)
        # streaming_client.sample(expansions="geo.place_id", tweet_fields="text", place_fields="country", threaded=True)

    async def send_senti(self, event):
        data = event['data']
        await self.send(json.dumps(data))