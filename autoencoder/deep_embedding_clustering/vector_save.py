import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from tqdm import tqdm
import numpy as np
import json
import argparse

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_path", required=True,
                 type=str, help="Please enter data path")
par.add_argument("-f", "--num_features", default=64, choices=[64, 256],
                 type=int, help="Set the feature size. (64/256)")
par.add_argument("-t", "--types", default="semantic", choices=["real", "semantic"],
                 type=str, help="Choose a data type. (real/semantic)")
par.add_argument("-c", "--number_of_classes", default=18,
                 type=int, help="number of classes")
args = par.parse_args()

## Set the parameters and data path
data_path = args.data_path + "/"
num_features = args.num_features
types = args.types
num_classes = args.number_of_classes

data_path = data_path + ("activity" if types == "real" else types)
data_all_path = data_path + "/image/all_dir_classifications"
log_paths = ["log/pth/best_loss_" + types + "_conv_autoencoder_d" + num_features + "_save_model.pth",
            "log/pth/best_nmi_" + types + "_dec_conv_d" + num_features + "_save_model.pth"]
data_names = ["conv_" + ("re" if types == "real" else "se") + "_d" + num_features,
             "dec_conv_" + ("re" if types == "real" else "se") + "_d" + num_features]

## Set data loader
dataset = datasets.ImageFolder(root=data_all_path,
                           transform=transforms.Compose([
                               transforms.Resize((256,128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5,0.5, 0.5)),
                           ]))

class AutoEncoderConv(nn.Module):
    def __init__(self, num_classes, num_features):
        super(AutoEncoderConv, self).__init__()
        self.num_features = num_features
        self.fc1 = nn.Linear(64*8*4, 2048)
        self.fc2 = nn.Linear(2048, 256)
        if self.num_features == 64:
            self.fc3 = nn.Linear(256, 64)
            self.de_fc1 = nn.Linear(64, 256)
        self.de_fc2 = nn.Linear(256, 2048)
        self.de_fc3 = nn.Linear(2048, 64*8*4)

        self.encoder = nn.Sequential(
            # Input : 3*256*128
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 8*128*64

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 16*64*32

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 16*32*16
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 32*16*8
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 64*8*4
        )

        self.decoder = nn.Sequential(
            # 64*8*4

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(True),
            # 32*16*8

            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(True),
            # 16*32*16
            
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.ReLU(True),
            # 16*64*32
            
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(True),
            # 8*128*64

            nn.ConvTranspose2d(8, 3, 2, stride=2),
            nn.ReLU(True)
            # 3*256*128 
        )
        
        self.alpha = 1.0
        self.clusterCenter = nn.Parameter(torch.zeros(num_classes,num_features))
        self.pretrainMode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                
    def setPretrain(self,mode):
        self.pretrainMode = mode
        
    def updateClusterCenter(self, cc):
        self.clusterCenter.data = torch.from_numpy(cc)
        
    def getTDistribution(self, x, clusterCenter):
        xe = torch.unsqueeze(x,1).cuda() - clusterCenter.cuda()
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t() #due to divison, we need to transpose q
        return q
        
    def forward(self, x):
        y = self.encoder(x)
        y = F.relu(self.fc1(y.view(y.size(0), -1)))
        y = F.relu(self.fc2(y))
        if self.num_features == 64:
            y = F.relu(self.fc3(y))
        y_e = y
        
        #if not in pretrain mode, we only need encoder
        if self.pretrainMode == False:
            return y, self.getTDistribution(y, self.clusterCenter)
        
        # -- decoder --
        if self.num_features == 64:
            y = F.relu(self.de_fc1(y))
        y = F.relu(self.de_fc2(y))
        y = F.relu(self.de_fc3(y))
        y_d = self.decoder(y.view(y.size(0), 64, 8, 4))
        return y_e, y_d

for log_path, data_name in zip(log_paths, data_names):
    model = AutoEncoderConv(num_classes, num_features).cuda()
    model.load_state_dict(torch.load(log_path))
    model.eval()

    name = []
    result = []
    name_dic = {}
    for data, img_path in tqdm(zip(dataset, dataset.imgs)):
        img = Variable(data[0]).cuda().unsqueeze(0)
        label = img_path[0].split('/')[-1].split('.')[0]
        output, predict = model(img)

        name.extend([label])
        result.extend(output.cpu().detach().tolist())

    result = np.array(result)

    save_path = "save_data/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + data_name + "/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(save_path + data_name + "_data.npy", result)
    name_dic["name"] = name
    name_json = json.dumps(name_dic)
    name_file = open(save_path + data_name + "_names.json","w")
    name_file.write(name_json)
    name_file.close()
