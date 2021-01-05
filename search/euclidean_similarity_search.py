from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import json
import argparse
from tqdm import tqdm

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='seq2seq_d64',
                 type=str, help="data name")
par.add_argument("-i", "--iamge_name", default='16019', # calendar activity
                 type=str, help="name of image to search")
args = par.parse_args()

data_add_names = json.loads(open("../data/" + args.data_name +
                 "/" + args.data_name + '_names.json').read())
data_add_data = np.load("../data/" + args.data_name +
                 "/" + args.data_name + '_data.npy')


print("data_add_data shape = {}".format(data_add_data.shape))

result = data_add_data
result_data = dict()
inp = data_add_names['name'].index(args.iamge_name)

for i in tqdm(range(len(result))):
    result_data[str(i)] = euclidean_distances(
            result[inp:inp+1], result[i:i+1]).tolist()[0][0]

r = sorted(result_data.items(), key=lambda x: x[1])

for i in range(0, 200, 1):
    print("TOP%d - Image name: %s, Distance: %0.3f" % (i, data_add_names['name'][int(r[i][0])], r[i][1]))


for i in tqdm(range(len(result))):
    if data_add_names['name'][int(r[i][0])] == '2':
        print("TOP%d - Image name: %s, Distance: %0.3f" % (i+1, data_add_names['name'][int(r[i][0])], r[i][1]))

for i in tqdm(range(len(result))):
    if data_add_names['name'][int(r[i][0])] == '9385':
        print("TOP%d - Image name: %s, Distance: %0.3f" % (i+1, data_add_names['name'][int(r[i][0])], r[i][1]))
