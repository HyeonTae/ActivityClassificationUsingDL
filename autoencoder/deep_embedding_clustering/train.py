import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
#import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import argparse

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_path", required=True,
                 type=str, help="Please enter data path")
par.add_argument("-f", "--num_features", default=64, choices=[64, 256],
                 type=int, help="Set the feature size. (64/256)")
par.add_argument("-t", "--types", default="semantic", choices=["real", "semantic"],
                 type=str, help="Choose a data type. (real/semantic)")
par.add_argument("-e", "--number_of_epochs", default=100,
                 type=int, help="number of epochs")
par.add_argument("-c", "--number_of_classes", default=18,
                 type=int, help="number of classes")
args = par.parse_args()

## Set the parameters and data path
data_path = args.data_path + "/"
num_features = args.num_features
types = args.types
num_epochs = args.number_of_epochs
num_classes = args.number_of_classes
learning_rate = 0.001
batch_size = 128

## Set each path
log_path = "log/pth/best_loss_" + types + "_conv_autoencoder_d" + num_features + "_save_model.pth"
data_path = data_path + ("activity" if types == "real" else types)
data_all_path = data_path + "/image/all"
data_train_path = data_path + "/image/train"
data_test_path = data_path + "/image/test"

if not os.path.exists("./log/"):
    os.mkdir("./log/")
if not os.path.exists("./log/pth/"):
    os.mkdir("./log/pth/")
save_log_path = "./log/check_point/"
save_img_path = "./log/img/"
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)
if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)
save_log_path = save_log_path + ("re" if types == "real" else "se") + "_d" + num_features
save_img_path = save_img_path + ("re" if types == "real" else "se") + "_d" + num_features
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)
if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)

## Set data loader
dataset = datasets.ImageFolder(root=data_all_path,
                           transform=transforms.Compose([
                               transforms.Resize((256,128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=8)

train_dataset = datasets.ImageFolder(root=data_train_path,
                           transform=transforms.Compose([
                               transforms.Resize((256,128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=8)

test_dataset = datasets.ImageFolder(root=data_test_path,
                           transform=transforms.Compose([
                               transforms.Resize((256,128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))
test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=8)

## Accuracy
nmi = normalized_mutual_info_score
def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

## Convolutional autoencoder
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
        # -- encoder --
        y = self.encoder(x)
        y = F.relu(self.fc1(y.view(y.size(0), -1)))
        y = F.relu(self.fc2(y))
        if self.num_features == 64:
            y = F.relu(self.fc3(y))
        y_e = y

        # if not in pretrain mode, we only need encoder
        if self.pretrainMode == False:
            return y, self.getTDistribution(y, self.clusterCenter)

        # -- decoder --
        if self.num_features == 64:
            y = F.relu(self.de_fc1(y))
        y = F.relu(self.de_fc2(y))
        y = F.relu(self.de_fc3(y))
        y_d = self.decoder(y.view(y.size(0), 64, 8, 4))
        return y_e, y_d

## Controlling the training process of DEC
class DEC:
    def __init__(self,n_clusters,n_features, alpha=1.0):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.alpha = alpha
        
    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)
    def logAccuracy(self,pred,label):
        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
          % (acc(label, pred), nmi(label, pred)))
    @staticmethod
    def kld(q,p):
        res = torch.sum(p*torch.log(p/q),dim=-1)
        return res
    
    def validateOnCompleteTestData(self,test_loader,model):
        model.eval()
        for i,d in enumerate(test_loader):
            if i == 0:
                to_eval = model(d[0].cuda())[0].data.cpu().numpy()
                true_labels = d[1].cpu().numpy()
            else:
                to_eval = np.concatenate((to_eval, model(d[0].cuda())[0].data.cpu().numpy()), axis=0)
                true_labels = np.concatenate((true_labels, d[1].cpu().numpy()), axis=0)

        #print("to_eval.shape : {}".format(to_eval.shape))
        #print("true_labels.shape : {}".format(true_labels.shape))
        #print("len(np.unique(true_labels) is {}".format(len(np.unique(true_labels))))
        
        km = KMeans(n_clusters=len(np.unique(true_labels)))
        y_pred = km.fit_predict(to_eval)
        
        return acc(true_labels, y_pred), nmi(true_labels, y_pred)
    
    def pretrain(self, dataloader, num_epochs):
        model = AutoEncoderConv(self.n_clusters, self.n_features).cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = 1.0
        best_epoch = 0

        for epoch in range(num_epochs):
            for data in tqdm(dataloader):
                img, _ = data
                img = Variable(img).cuda()
                # ===================forward=====================
                _, output = model(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            if epoch % 2 == 0:
                save_image(img, save_img_path + "image_{}.png".format(epoch))
                save_image(output, save_img_path + "g_image_{}.png".format(epoch))

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch+1
                torch.save(model.state_dict(), "./log/pth/best_loss_"
                           + types + "_conv_autoencoder_d" + num_features + "_save_model.pth")

            print("epoch [{}/{}], loss:{:.4f}, best loss:{:.4f}[{}/{}]"
                  .format(epoch+1, num_epochs, loss.item(), best_loss, best_epoch, num_epochs))
            with open(save_log_path + "epoch" + str(epoch+1), 'w') as f:
                f.write("epoch [{}/{}], loss:{:.4f}, best loss:{:.4f}[{}/{}]"
                  .format(epoch+1, num_epochs, loss.item(), best_loss, best_epoch, num_epochs))
            torch.save(model.state_dict(), "./log/pth/"
                       + types + "_conv_autoencoder_d" + num_features + "_save_model.pth")

        torch.save(model.state_dict(), "./log/pth/"
                   + types + "_conv_autoencoder_d" + num_features + "_save_model.pth")
        
    def clustering(self, mbk, x, model):
        model.eval()
        y_pred_ae,_ = model(x)
        y_pred_ae = y_pred_ae.data.cpu().numpy()
        y_pred = mbk.partial_fit(y_pred_ae) # seems we can only get a centre from batch
        self.cluster_centers = mbk.cluster_centers_ # keep the cluster centers
        model.updateClusterCenter(self.cluster_centers)
    def train(self,train_loader, test_loader, num_epochs):
        # this method will start training for DEC cluster
        best_epoch = 0
        best_nmi = 0.0
        model = AutoEncoderConv(self.n_clusters, self.n_features).cuda()
        model.load_state_dict(torch.load(log_path))
        model.setPretrain(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print('Initializing cluster center with pre-trained weights')
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=batch_size)
        got_cluster_center = False
        for epoch in range(num_epochs):
            for data in tqdm(train_loader):
                img, _ = data
                img = Variable(img).cuda()
                optimizer.zero_grad()
                # step 1 - get cluster center from batch
                # here we are using minibatch kmeans to be able to cope with larger dataset.
                if not got_cluster_center:
                    self.clustering(mbk, img, model)
                    if epoch > 1:
                        got_cluster_center = True
                else:
                    model.train()
                    # now we start training with acquired cluster center
                    feature_pred,q = model(img)
                    # get target distribution
                    p = self.target_distribution(q)
                    kld_loss = self.kld(q,p).mean()
                    kld_loss.backward()
                    optimizer.step()
            
            if got_cluster_center:
                acc, nmi = self.validateOnCompleteTestData(test_loader,model)
                if best_nmi < nmi:
                    best_epoch = epoch+1
                    best_nmi = nmi
                    torch.save(model.state_dict(), "./log/pth/best_nmi_"
                               + types + "_dec_conv_d" + num_features + "_save_model.pth")

                print('epoch [{}/{}], loss:{:.4f}, acc:{:.4f}, nmi:{:.4f}, *best_nmi:{:.4f}[{}/{}]'
                      .format(epoch+1, num_epochs, kld_loss.item(), acc, nmi, best_nmi, best_epoch, num_epochs))
                with open(save_log_path + "dec_epoch" + str(epoch+1), 'w') as f:
                    f.write("epoch [{}/{}], loss:{:.4f}, acc:{:.4f}, nmi:{:.4f}, *best_nmi:{:.4f}[{}/{}]"
                      .format(epoch+1, num_epochs, kld_loss.item(), acc, nmi, best_nmi, best_epoch, num_epochs))
                torch.save(model.state_dict(), "./log/pth/"
                           + types + "_dec_conv_d" + num_features + "_save_model.pth")

if __name__ == "__main__":
    dec = DEC(num_classes, num_features)
    dec.pretrain(dataloader, num_epochs)
    dec.train(train_loader, test_loader, num_epochs)
