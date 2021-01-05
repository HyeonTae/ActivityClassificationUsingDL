import numpy as np
from sklearn.neighbors import BallTree
import argparse
from tqdm import tqdm
import os
import json

par = argparse.ArgumentParser()
par.add_argument("-i", "--image_name", default='16019', # calendar activity
                 type=str, help="name of image to search")
args = par.parse_args()

#data_name_list = ['conv_re_gray_d256', 'conv_re_gray_d64', 'conv_re_gray_d64_2',
#                  'conv_re_RGB_d256', 'conv_re_RGB_d64', 'conv_re_RGB_d64_2',
#                  'conv_se_gray_d256', 'conv_se_gray_d64', 'conv_se_gray_d64_2',
#                  'conv_se_RGB_d256', 'conv_se_RGB_d64', 'conv_se_RGB_d64_2',
#                  'conv_se_RGB_rico', 'seq2seq_d256', 'seq2seq_d64', 'rico_d64']

re_d1 = ['conv_re_gray_d64', 'conv_re_gray_d64_2', 'conv_re_RGB_d64', 'conv_re_RGB_d64_2']
se_d1 = ['conv_se_gray_d64', 'conv_se_gray_d64_2', 'conv_se_RGB_d64', 'conv_se_RGB_d64_2']
re_d2 = ['seq2seq_d256']
se_d2 = ['seq2seq_d64']


for data_name in tqdm(data_name_list):
    data_names = json.loads(open("../data/" + data_name +
                     "/" + data_name + '_names.json').read())
    data_data = np.load("../data/" + data_name +
                     "/" + data_name + '_data.npy')

    img_path = "/home/hyeontae/data/ActivityClustering/rico_data/activity/image/all/1/"
    save_path = "./result/" + data_name + "/"

    result = data_data
    inp = data_names['name'].index(args.image_name)

    tree = BallTree(result)

    dist, ind = tree.query([result[inp]], 200)

    save_path = "./result/" + data_name + "/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for i in range(len(dist[0])):
        os.system("cp " + img_path + data_names['name'][ind[0][i]] + ".jpg "
                + save_path +  str(i) + "_" + data_names['name'][ind[0][i]] + ".jpg")
