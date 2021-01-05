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

data_names = json.loads(open("../data/" + args.data_name +
                 "/" + args.data_name + '_names.json').read())
data_data = np.load("../data/" + args.data_name +
                 "/" + args.data_name + '_data.npy')

img_path = "/home/hyeontae/data/ActivityClustering/rico_data/activity/image/all/1/"

result = data_data
inp = data_names['name'].index(args.iamge_name)

tree = BallTree(result)

dist, ind = tree.query([result[inp]], 200)

save_path = "./result/" + args.data_name + "/"
if not os.path.isdir(save_path):
    os.mkdir(save_path)

for i in range(len(dist[0])):
    os.system("cp " + img_path + data_names['name'][ind[0][i]] + ".jpg "
            + save_path +  str(i) + "_" + data_names['name'][ind[0][i]] + ".jpg")
