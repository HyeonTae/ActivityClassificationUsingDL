import os
import argparse
import logging
import sys
import json

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

from models.trainer import Trainer
from models.seq2seq import Seq2seq
from loss.loss import Perplexity
from dataset import fields

import matplotlib.pyplot as plt

log_level = 'info'
LOG_FORMAT = '%(asctime)s %(levelname)-6s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, log_level.upper()))

train_path = "data/train_data.txt"
dev_path = "data/val_data.txt"
config_path = "models/config.json"

'''
config = { "max_len": 150,
           "embedding_size": 14,
           "hidden_size": 64,
           "input_dropout_p": 0,
           "dropout_p": 0,
           "n_layers": 1,
           "bidirectional": False,
           "rnn_cell": "lstm",
           "embedding": None,
           "update_embedding": True,
           "get_context_vector": False,
           "use_attention": true }
'''

optimizer = "Adam"
seq2seq = None
config_json = open(config_path).read()
config = json.loads(config_json)
print(json.dumps(config, indent=4))

max_len = config["max_len"]

src = fields.SourceField()
srcp = fields.SourceField()
tgt = fields.TargetField()
tgtp = fields.TargetField()

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len
train = torchtext.data.TabularDataset(
    path=train_path, format='tsv',
    fields=[('src', src), ('srcp', srcp), ('tgt', tgt), ('tgtp', tgtp)],
    filter_pred=len_filter
)
dev = torchtext.data.TabularDataset(
    path=dev_path, format='tsv',
    fields=[('src', src), ('srcp', srcp), ('tgt', tgt), ('tgtp', tgtp)],
    filter_pred=len_filter
)
src.build_vocab(train)
tgt.build_vocab(train)
srcp.build_vocab(train)
tgtp.build_vocab(train)

input_vocab = src.vocab
output_vocab = tgt.vocab

weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

#print("src vocab size = %d" % (len(src.vocab)))
#print("tat vacab size = %d" % (len(tgt.vocab)))
#print("srcp vocab size = %d" % (len(srcp.vocab)))
#print("tatp vacab size = %d" % (len(tgtp.vocab)))

seq2seq = Seq2seq(config, len(src.vocab), len(tgt.vocab), tgt.sos_id, tgt.eos_id)

if torch.cuda.is_available():
    seq2seq.cuda()

for param in seq2seq.parameters():
       param.data.uniform_(-0.08, 0.08)

# train
t = Trainer(loss=loss, batch_size=32,
            checkpoint_every=50,
            print_every=100,
            hidden_size=config["hidden_size"],
            path="rico_data",
            file_name="autoencoder_128_h64",
            early_stop=50)

seq2seq, ave_loss, character_accuracy_list, sentence_accuracy_list, f1_score_list = t.train(seq2seq, train,
                                                                             num_epochs=60, dev_data=dev,
                                                                             optimizer=optimizer,
                                                                             teacher_forcing_ratio=0.5)

