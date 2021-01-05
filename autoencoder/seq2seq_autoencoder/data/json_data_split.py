import glob
from tqdm import tqdm
import os
import random

test = list()
all = list()
temp = list()

path = "data_processing/semantic_annotations/json/"
ground_path = "ground_truth/"
save_path = "seq2seq_autoencoder/data/json_data/"

if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(save_path+"test"):
    os.mkdir(save_path+"test")
if not os.path.isdir(save_path+"train"):
    os.mkdir(save_path+"train")
if not os.path.isdir(save_path+"val"):
    os.mkdir(save_path+"val")

test = list()
for dir in tqdm(glob.glob(ground_path+"*")):
    for f in glob.glob(dir+"/*"):
        test.append(f.split("/")[-1].split(".")[0]+".json")

for dir in tqdm(glob.glob(path+"*")):
    all.append(dir.split("/")[-1])

print("test json data...")
for t in tqdm(all):
    if t not in test:
        temp.append(t)
    else:
        os.system("cp " + path+t + " " + save_path+"test/")

random.shuffle(temp)
m = int(len(temp)*4/5)
train = temp[:m]
val = temp[m:]

print("train json data...")
for i in tqdm(train):
    os.system("cp " + path+i + " " + save_path+"train/")

print("val json data...")
for i in tqdm(val):
    os.system("cp " + path+i + " " + save_path+"val/")

print("done...")
