import numpy as np 
from model import *
import torch 
import re 
class SentimentPredictor:
    def __init__(self, model_path = 'ckpt/sentiment_model.pt', vocab_path = 'vocab.txt', device = 'cpu'):
        self.device = device 
        self.vocab = self.read_txt_to_dict(vocab_path)
        self.vocab_size = len(self.vocab) + 1 
        self.embedding_dim = 64 
        self.output_dim = 1 
        self.hidden_dim = 256 
        self.model = self.load_model(model_path)
        self.model.to(self.device)
    
    @staticmethod
    def read_txt_to_dict(file_path): 
        result_dict = {}
        with open(file_path, 'r') as file: 
            for line in file: 
                key, value = line.strip().split(": ")
                result_dict[key] = int(value)
        return result_dict
    
    def load_model(self, model_path): 
        model = SentimentRNN(no_layers=2, vocab_size=self.vocab_size, hidden_dim=self.hidden_dim,
                             embedding_dim=self.embedding_dim, output_dim=self.output_dim, drop_prob=0.5)
        model.load_state_dict(torch.load(model_path, map_location = self.device))
        model.eval()
        return model 
    
    def preprocess_string(self, s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with no space
        s = re.sub(r"\s+", '', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)
        return s

    def padding_(self, sentences, seq_len):
        features = np.zeros((len(sentences), seq_len),dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features
    
    def predict_text(self, text): 
        word_seq = np.array([self.vocab[self.preprocess_string(word)] for word in text.split()
                             if self.preprocess_string(word) in self.vocab.keys()])
        word_seq = np.expand_dims(word_seq, axis=0)
        pad = torch.from_numpy(self.padding_(word_seq, 500))
        inputs = pad.to(self.device)
        batch_size = 1
        h = self.model.init_hidden(batch_size)
        h = tuple([each.data.to(self.device) for each in h])
        output, h = self.model(inputs, h)
        return output.item()
    
    def predict_sentiment(self, text): 
        pro = self.predict_text(text)
        status = "positive" if pro > 0.5 else "negative"
        pro = ( 1 - pro ) if status == "negative" else pro 
        return status, pro 


if __name__ == "__main__":
    sentiment_predictor = SentimentPredictor()
    string ="The film breathed new life into the genre, offering a fresh perspective and creative storytelling."
    status, pro = sentiment_predictor.predict_sentiment(string)
    print(f'Predicted sentiment is {status} with a probability of {pro:.4f}')