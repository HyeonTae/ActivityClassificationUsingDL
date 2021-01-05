import glob
import argparse
import copy
from sklearn.metrics.cluster import  adjusted_rand_score, normalized_mutual_info_score

par = argparse.ArgumentParser()
par.add_argument("-d1", "--data_name_1", default="conv",
                 type=str, help="data name 1")
par.add_argument("-d2", "--data_name_2", default=None,
                 type=str, help="data name 2")
par.add_argument("-t", "--fusion_type", default=None, choices=["add", "cat"],
                 type=str, help="fusion type(add/cat)")
par.add_argument("-s", "--scaler", default=None,
                 type=str, help="Select the scaler")
par.add_argument("-w", "--weight", default=None,
                 type=str, help="rico weight")
par.add_argument("-e", "--evaluation", default="nmi", choices=["purity", "nmi", "ari"],
                 type=str, help="Select the evaluation method(purity, nmi, ari)")
args = par.parse_args()

#clustering_algorithms = ['gaussian_mixture', 'dbscan', 'optics', 'birch', 'kmeans']
clustering_algorithms = ['gaussian_mixture', 'kmeans']
ground_truth_path = "../ground_truth/"

fusion_type = "_" + args.fusion_type if args.fusion_type is not None else ""
scaler = "_" + args.scaler if args.scaler is not None else ""
weight = "_" + args.weight if args.weight is not None else ""
data_2 = "_" + args.data_name_2 if args.data_name_2 is not None else ""
data_name = args.data_name_1 + data_2 + fusion_type + scaler + weight
print("====> " + data_name)

def Purity(ground_truth, clustering_result):
    g_result = dict()
    _ground_truth = copy.deepcopy(ground_truth)
    for c_key in clustering_result.keys():
        comp = list()
        for g_key in _ground_truth.keys():
            comp.append((g_key, c_key, len(list(
                set(ground_truth[g_key]).intersection(clustering_result[c_key]))),
                len(ground_truth[g_key])))
        comp.sort(key=lambda element:element[2])
        if len(_ground_truth.keys()) == 0:
            break
        g_result[comp[-1][0]] = list()
        g_result[comp[-1][0]].append(comp[-1][1])
        g_result[comp[-1][0]].append(comp[-1][2])
        del _ground_truth[comp[-1][0]]

    result = list()
    for g_key in ground_truth.keys():
        if g_key in g_result.keys():
            result.append((g_key, g_result[g_key][0], g_result[g_key][1], len(ground_truth[g_key])))
        else:
            result.append((g_key, "", 0, len(ground_truth[g_key])))

    matched = 0
    total = 0
    print(data_name + " " + clustering_algorithm)
    print("-----------------------------")
    for t in result:
        print("%.1f" % (t[2]/t[3]*100) + "%")
        matched += t[2]
        total += t[3]
    print("%.1f" % (matched/total*100) + "%\n")

def NMI(ground_truth_labels, ground_truth, clustering_result):
    clustering_resul_labels = copy.deepcopy(ground_truth_labels)
    for g_key in ground_truth.keys():
        for g in ground_truth[g_key]:
            ground_truth_labels[ground_truth_labels.index(g)] = g_key

    for c_key in clustering_result.keys():
        for c in clustering_result[c_key]:
            clustering_resul_labels[clustering_resul_labels.index(c)] = c_key

    print("%.3f" % (normalized_mutual_info_score(ground_truth_labels, clustering_resul_labels)))

def ARI(ground_truth_labels, ground_truth, clustering_result):
    clustering_resul_labels = copy.deepcopy(ground_truth_labels)
    for g_key in ground_truth.keys():
        for g in ground_truth[g_key]:
            ground_truth_labels[ground_truth_labels.index(g)] = g_key

    for c_key in clustering_result.keys():
        for c in clustering_result[c_key]:
            clustering_resul_labels[clustering_resul_labels.index(c)] = c_key

    print("%.3f" % (adjusted_rand_score(ground_truth_labels, clustering_resul_labels)))

def get_list(ground_truth):
    print("List Name")
    for t in ground_truth.keys():
        print(t)
    print("\n")

if __name__=="__main__":
    for i, clustering_algorithm in enumerate(clustering_algorithms):
        result_path = '../clustering/result/' + data_name + "/" + clustering_algorithm + "/"

        ground_truth = dict()
        ground_truth_temp = dict()
        ground_truth_labels = list()

        for dirs in glob.glob(ground_truth_path+"*"):
            ground_truth_temp[dirs.split("/")[-1]] = list()
            for files in glob.glob(dirs+"/*"):
                ground_truth_temp[dirs.split("/")[-1]].append(files.split("/")[-1].split(".")[0])
                ground_truth_labels.append(files.split("/")[-1].split(".")[0])

        ground_truth_list = list(ground_truth_temp.keys())
        ground_truth_list.sort()
        for g_key in ground_truth_list:
            ground_truth[g_key] = ground_truth_temp[g_key]

        clustering_result = dict()
        clustering_result_temp = dict()

        for dirs in glob.glob(result_path+"*"):
            with open(dirs, 'r') as f:
                res = f.read()
            clustering_result_temp[dirs.split("/")[-1]] = " ".join(res.split('\n')).split()

        clustering_result_list = list(clustering_result_temp.keys())
        clustering_result_list.sort()
        for c_key in clustering_result_list:
            clustering_result[c_key] = clustering_result_temp[c_key]

        if i == 0:
            get_list(ground_truth)

        if args.evaluation == "purity":
            Purity(ground_truth, clustering_result)
        elif args.evaluation == "nmi":
            NMI(ground_truth_labels, ground_truth, clustering_result)
        elif args.evaluation == "ari":
            ARI(ground_truth_labels, ground_truth, clustering_result)
