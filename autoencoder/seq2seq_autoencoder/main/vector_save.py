import os
import logging
import sys
import json
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

from models.trainer import Trainer
from models.seq2seq import Seq2seq
from loss.loss import Perplexity
from optim.optim import Optimizer
from dataset import fields
from evaluator.predictor import Predictor

import numpy as np

train_path = "data/train_data.txt"
log_path = "log/pth/rico_data_autoencoder_128_h64"
config_path = "models/config.json"

'''
config = { "max_len": 150,
           "embedding_size": 16,
           "hidden_size": 50,
           "input_dropout_p": 0,
           "dropout_p": 0,
           "n_layers": 1,
           "bidirectional": True,
           "rnn_cell": "lstm",
           "embedding": None,
           "update_embedding": False,
           "get_context_vector": False,
           "use_attention": true }
'''

optimizer = "Adam"
seq2seq = None
config_json = open(config_path).read()
config = json.loads(config_json)
print(json.dumps(config, indent=4))

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, "info".upper()))

# Prepare dataset
src = fields.SourceField()
tgt = fields.TargetField()
srcp = fields.SourceField()
tgtp = fields.TargetField()
max_len = config["max_len"]
def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len
train = torchtext.data.TabularDataset(
    path=train_path, format='tsv',
    fields=[('src', src), ('srcp', srcp), ('tgt', tgt), ('tgtp', tgtp)],
    filter_pred=len_filter
)

src.build_vocab(train)
tgt.build_vocab(train)
srcp.build_vocab(train)
tgtp.build_vocab(train)
input_vocab = src.vocab
output_vocab = tgt.vocab
input_part_vocab = srcp.vocab
output_part_vocab = tgtp.vocab

# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

# Initialize model
seq2seq = Seq2seq(config, len(src.vocab), len(tgt.vocab), tgt.sos_id, tgt.eos_id)
if torch.cuda.is_available():
    seq2seq.cuda()

for param in seq2seq.parameters():
    param.data.uniform_(-0.08, 0.08)

seq2seq.load_state_dict(torch.load(log_path))
seq2seq.eval()

predictor = Predictor(seq2seq, input_vocab, input_part_vocab,
                      output_vocab, output_part_vocab)
# Dataset Load
lines = open('data/all_data.txt').read().strip().split('\n')
pairs = [[s for s in l.split('\t')] for l in lines]

# Predict
try:
    rsult = []
    c = 0
    name_dic = {}
    name = []
    over_data = []
    for pair in tqdm(pairs):
        name.append(pair[0])
        seq = pair[1].split(" ")
        if len(seq) > 150:
            over_data.append(pair[0])
        partition = pair[2].split(" ")
        tgt_seq, tgt_att_list, encoder_outputs = predictor.predict(seq,partition)
        if pair[1] == " ".join(tgt_seq).replace(" <eos>", ""):
            c += 1

        data = encoder_outputs[len(encoder_outputs)-1].tolist()
        rsult.append(data)

    rsult = np.array(rsult)
    print("accuracy : %f" % (c/len(pairs)))
    np.save('../data/seq2seq/seq2seq_data.npy', rsult)
    name_dic["name"] = name
    name_json = json.dumps(name_dic)
    name_file = open("../data/seq2seq/seq2seq_names.json","w")
    name_file.write(name_json)
    name_file.close()

except KeyboardInterrupt:
    pass
