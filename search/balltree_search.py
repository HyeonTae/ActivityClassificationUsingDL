import numpy as np
from sklearn.neighbors import BallTree
import argparse
from tqdm import tqdm
import os
import json

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='rico',
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
inp = data_add_names['name'].index(args.iamge_name)

tree = BallTree(result)

dist, ind = tree.query([result[inp]], 100)

#print(dist[0])
#print(ind[0])
#for i in ind[0]:
#    print(type(int(i)))
#for i in ind[0]:
#    print(data_add_names['name'][i])


for i in range(len(dist[0])):
    print("TOP%d - Image name: %s, Distance: %0.3f" % (i, data_add_names['name'][ind[0][i]], dist[0][i]))
