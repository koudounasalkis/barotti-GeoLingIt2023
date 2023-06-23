import os
import torch.utils.data as data
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from random import randint
import numpy as np
import glob
import pandas as pd
import re

"""
    This class builds a pretraining dataset in which we can sample positive and negative pairs:
    Specifically, given an anchor tweet:
        - a positive pair is built by changing the dialect words with other words of the same region with a certain probability p
        - a negative pair is built by changing the dialect words with other words of a different region with a certain probability p
"""
class PretrainingSentenceDataset(data.Dataset):
    def __init__(self, dialect_words, dialect_labels, model_path, italian_words, tweet_dataset, max_length=512):
        self.dialect_words = dialect_words
        self.dialect_labels = dialect_labels
        self.italian_words = italian_words
        self.tweet_dataset = pd.read_csv(tweet_dataset, sep="\t")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.proportion = [0.5, 0.5]

    def __len__(self):
        return len(self.tweet_dataset)
    
    def replace_words(self, tweet, region):
        """
        Replace the words in the tweet with a certain probability
        """
        tweet = tweet.split()
        random_choice = np.random.choice([0, 1], p=self.proportion)

        if random_choice == 0:
            contrastive_label = torch.tensor(-1)
        else:
            contrastive_label = torch.tensor(1)

        for i, word in enumerate(tweet):
            if word.lower().strip() not in self.italian_words and word not in ["USER", "LOCATION", "URL"]:
                new_word = self.sample_new_words(word, contrastive_label, region)
                tweet[i] = new_word

        return " ".join(tweet), contrastive_label
    

    def sample_new_words(self, base_word, contrastive_label, region):
        """
        Sample a new word from the dialect words list
        """
        region = region.lower().replace(" ", "").replace("-","").replace("'","")

        if contrastive_label == torch.tensor(-1):
            sampling_list = self.dialect_words[self.dialect_labels != region]
        else:
            sampling_list = self.dialect_words[self.dialect_labels == region]

        new_word = np.random.choice(sampling_list)
        while new_word == base_word:
            new_word = np.random.choice(sampling_list)
        return new_word


    def __getitem__(self, index):
        item = self.tweet_dataset.iloc[index]
        base_tweet = item["text"]
        base_tweet = re.sub(r'[^\w\s]', '', base_tweet)
        region = item["region"]
        contrastive_tweet, contrastive_label = self.replace_words(base_tweet, region)
       
        ## Tokenize the dialects words
        input_features = self.tokenizer(
            base_tweet,
            return_tensors="pt",
            truncation=True,
            verbose=False,
            padding="max_length")
        input_features_2 = self.tokenizer(
            contrastive_tweet,
            return_tensors="pt",
            truncation=True,
            verbose=False,
            padding="max_length")

        ## Remove the batch dimension
        for k,v in input_features.items():
            input_features[k] = v.squeeze(0)
            input_features_2[k] = v.squeeze(0)
            
        inputs = {
            "input_ids": torch.cat((input_features["input_ids"], input_features_2["input_ids"]), dim=0),
            "attention_mask": torch.cat((input_features["attention_mask"], input_features_2["attention_mask"]), dim=0),
            "labels": contrastive_label
        }
        return inputs