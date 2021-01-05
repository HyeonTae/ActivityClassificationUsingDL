import numpy as np
import json
import argparse
import os
from tqdm import tqdm
import glob

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='rico',
                 type=str, help="data name")
args = par.parse_args()

data_names = json.loads(open(args.data_name + "/" +
        args.data_name + '_names.json').read())
data = np.load(args.data_name + "/" +
        args.data_name + '_data.npy')

ground_path = "../ground_truth/"

test_names = list()
for dir in tqdm(glob.glob(ground_path+"*")):
    for f in glob.glob(dir+"/*"):
        test_names.append(f.split("/")[-1].split(".")[0])

test_index = list()
for t in test_names:
    if t in data_names['name']:
        test_index.append(data_names['name'].index(t))

test_data = list()
for i in test_index:
    test_data.append(data[i])

test_data_np = np.array(test_data)

name_dic = dict()
np.save(args.data_name + "/" +
        args.data_name + '_test_data.npy', test_data_np)
name_dic["name"] = test_names
name_json = json.dumps(name_dic)
name_file = open(args.data_name + "/" +
        args.data_name + '_test_names.json',"w")
name_file.write(name_json)
name_file.close()

os.system('rm ' + args.data_name + "/" +
        args.data_name + '_names.json')
os.system('rm ' + args.data_name + "/" +
        args.data_name + '_data.npy')
os.system('mv ' + args.data_name + "/" +
        args.data_name + '_test_names.json ' + args.data_name + "/" +
        args.data_name + '_names.json')
os.system('mv ' + args.data_name + "/" +
        args.data_name + '_test_data.npy ' + args.data_name + "/" +
        args.data_name + '_data.npy')
