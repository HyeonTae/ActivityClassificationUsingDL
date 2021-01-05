import json
import argparse
import os
from tqdm import tqdm

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='seq2seq_d64',
                 type=str, help="data name")
par.add_argument("-t", "--threshold",
                 type=float, help="Set threshold")
args = par.parse_args()

data_path = ("result/" + args.data_name  + "/" + args.data_name +
             "_t" + str(args.threshold) + ".json")

image_path = "/home/hyeontae/data/ActivityClustering/rico_data/activity/image/all/1/"
save_path = "classification_result/"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
save_path += args.data_name
if not os.path.isdir(save_path):
    os.mkdir(save_path)

jsonData = json.loads(open(data_path).read())
print("Number of class : {}".format(len(jsonData)))
print("Number of etc : {}".format(len(jsonData['etc'])))

data_sort = sorted(jsonData.items(), key=lambda item: len(item[1]), reverse = True)

total = 0
cnt = 1
for d in tqdm(data_sort):
    if d[0] == 'etc':
        continue
    elif len(d[1]) > 500:
        continue
    elif len(d[1]) > 30:
        total += len(d[1])
        save_p = save_path+"/"+str(cnt)
        if not os.path.isdir(save_p):
            os.mkdir(save_p)
        for i in d[1]:
            os.system("cp " + image_path + i + ".jpg " + save_p)
        cnt += 1

print("Total = {}".format(total))
