import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import random as rnfv

from keras.utils import to_categorical 
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
from torch import nn
from torch_snippets import *
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BigBirdModel, LongformerModel, LongformerTokenizer, BertTokenizer
import warnings
warnings.filterwarnings('ignore')

class Logger:
    def __init__(self):
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def record(self, pos, train_loss=None, train_acc=None, val_loss=None, val_acc=None, end='\n'):
        train_loss_str = f'{train_loss:.4f}' if train_loss is not None else 'N/A'
        train_acc_str = f'{train_acc:.4f}' if train_acc is not None else 'N/A'
        val_loss_str = f'{val_loss:.4f}' if val_loss is not None else 'N/A'
        val_acc_str = f'{val_acc:.4f}' if val_acc is not None else 'N/A'

        if train_loss is not None:
            self.train_loss_list.append(train_loss)
        if train_acc is not None:
            self.train_acc_list.append(train_acc)
        if val_loss is not None:
            self.val_loss_list.append(val_loss)
        if val_acc is not None:
            self.val_acc_list.append(val_acc)
        
        print(f"Position: {pos:.4f}, Train Loss: {train_loss_str}, Train Acc: {train_acc_str}, Val Loss: {val_loss_str}, Val Acc: {val_acc_str}", end=end)

    def report_avgs(self, epoch):
        avg_train_loss = sum([x.item() for x in self.train_loss_list]) / len(self.train_loss_list) if self.train_loss_list else 0.0
        avg_train_acc = sum([x.item() for x in self.train_acc_list]) / len(self.train_acc_list) if self.train_acc_list else 0.0
        avg_val_loss = sum([x.item() for x in self.val_loss_list]) / len(self.val_loss_list) if self.val_loss_list else 0.0
        avg_val_acc = sum([x.item() for x in self.val_acc_list]) / len(self.val_acc_list) if self.val_acc_list else 0.0

        print(f"\nEpoch {epoch:.4f} - Avg Train Loss: {avg_train_loss:.4f}, Avg Train Acc: {avg_train_acc:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Avg Val Acc: {avg_val_acc:.4f}")

    def plot_epochs(self, num_epochs = 10):
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        for i in range(1, num_epochs+1):
            train_len = len(self.train_loss_list) // num_epochs
            val_len = len(self.val_loss_list) // num_epochs

            train_loss_list.append(np.array([elem.item() for elem in self.train_loss_list[train_len*(i - 1):train_len*i]]).mean())
            train_acc_list.append(np.array([elem.item() for elem in self.train_acc_list[train_len*(i - 1):train_len*i]]).mean())
            val_loss_list.append(np.array([elem.item() for elem in self.val_loss_list[val_len*(i - 1):val_len*i]]).mean())
            val_acc_list.append(np.array([elem.item() for elem in self.val_acc_list[val_len*(i - 1):val_len*i]]).mean())
            epochs = range(1, len(train_loss_list) + 1)

        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_list, 'b', label='Training loss')
        plt.plot(epochs, val_loss_list, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc_list, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc_list, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Model
class Bert_Aggression_Identification_Model(nn.Module):
    def __init__(self, h1, h2, h3, h4, class_num, drop_out_rate=0.5):
        super(Bert_Aggression_Identification_Model, self).__init__()
        self.bert = BigBirdModel.from_pretrained('google/bigbird-roberta-base', attention_type = "original_full", )

        self.dropout1 = nn.Dropout(drop_out_rate)
        self.dropout2 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(h1, h2)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(h2, h3)
        self.LeakyRelu = nn.LeakyReLU()
        self.linear3 = nn.Linear(h3, h4)
        self.linear4 = nn.Linear(h4, class_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, tokens, masks):
        pooled_output = self.bert(tokens, attention_mask=masks, output_hidden_states=True)
        d = self.dropout1(pooled_output[0][:, 0, :])
        x = self.linear1(d)
        x = self.LeakyRelu(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.LeakyRelu(x)
        x = self.linear4(x)
        proba = self.softmax(x)

        return proba
# Data

class BigBirdData(Dataset):
    def __init__(self, X, y, word_max_len):
        super().__init__()
        self.X = X
        self.y = y
        self.tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
        tokens = list(map(lambda t: ['[CLS]'] + self.tokenizer.tokenize(t) + ['[SEP]'], self.X))
        self.tokens_ids = pad_sequences(list(map(self.tokenizer.convert_tokens_to_ids, tokens)),
                                   maxlen=word_max_len, truncating="post", padding="post", dtype="int")
        self.masks = [[float(i > 0) for i in ii] for ii in self.tokens_ids]

        print('Token ids size:', self.tokens_ids.shape)
        print('Masks size:', np.array(self.masks).shape)
        print('y size:', np.array(self.y).shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        tokens_id = self.tokens_ids[ind]
        label = self.y[ind]
        mask = self.masks[ind]
        return tokens_id, label, mask

    def collate_fn(self, data):
        tokens_ids, labels, masks = zip(*data)
        tokens_ids = torch.tensor(tokens_ids).to(device)
        labels = torch.tensor(labels).float().to(device)
        masks = torch.tensor(masks).to(device)
        return tokens_ids, labels, masks

    def choose(self):
        return self[np.random.randint(len(self))]
    
# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_list,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(torch.tensor(y_pred), torch.tensor(y_true))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_list, yticklabels=class_list,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax