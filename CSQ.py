# VTS (CSQ with ViT Backbone - ICME 2022)
# paper [Vision Transformer Hashing for Image Retrieval, ICME 2022](https://arxiv.org/pdf/2109.12564.pdf)
# CSQ basecode considered from https://github.com/swuxyj/DeepHash-pytorch

from utils.tools import *
from network import *
from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
from SRetModel.SReT import SReT

import argparse
import os
import random
import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.tensorboard import SummaryWriter
summaryWriter = SummaryWriter("./logs/")



SReT_T = SReT(
    image_size=224,
    patch_size=16,
    stride=8,
    base_dims=[32, 32, 32],
    depth=[4, 10, 6],
    recursive_num=[2,5,3],
    heads=[2, 4, 8],
    groups1=[8, 4, 1],
    groups2=[2, 1, 1],
    mlp_ratio=3.6,
    np_mlp_ratio=1,
    hash_length=64
)


    # if pretrained:
    #     state_dict = \
    #     torch.load('SReT_S.pth', map_location='cpu')
    #     model.load_state_dict(state_dict['model'])
    # return model

def get_config():
    config = {
        "dataset": "cifar10",        
        #"net": AlexNet, "net_print": "AlexNet",
        #"net":ResNet, "net_print": "ResNet",
        # "net": VisionTransformer, "net_print": "ViT-B_32", "model_type": "ViT-B_32", "pretrained_dir": "/home/d2070/wjy/VisionTransformerHashing-main/pretrainedViT/ViT-B_32.npz",
        #"net": VisionTransformer, "net_print": "ViT-B_16", "model_type": "ViT-B_16", "pretrained_dir": "pretrainedVIT/ViT-B_16.npz",
        "net": SReT_T, "net_print": "SReT_T_64", "model_type": "SReT_T_64", "pretrained_dir": "pretrainedSReT/SReT_T.pth",        
        "bit_list": [64],
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "device": torch.device("cuda"), "save_path": "Checkpoints_Results",
        "epoch": 200, "test_map": 30, "batch_size": 32, "resize_size": 256, "crop_size": 224,
        "info": "CSQ + cls", "lambda": 0.0001,
    }
    config = config_dataset(config)
    return config



def train_val(config, bit):
    start_epoch = 1
    Best_mAP = 0
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    
    num_classes = config["n_class"]
    hash_bit = bit
    
    net = config["net"].to(device)
    
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    best_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-BestModel.pt")
    trained_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-IntermediateModel.pt")
    results_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + ".txt")
    f = open(results_path, 'a')
    
    # if os.path.exists(trained_path):
    #     print('==> Resuming from checkpoint..')
    #     checkpoint = torch.load(trained_path)
    #     net.load_state_dict(checkpoint['net'])
    #     Best_mAP = checkpoint['Best_mAP']
    #     start_epoch = checkpoint['epoch'] + 1
    # else:
    # if "ViT" in config["net_print"]:
    # Load from pretrained
    print('==> Loading from pretrained model..')
    net.load_state_dict(torch.load(config['pretrained_dir'])['model'],strict=False)
    
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = CSQLoss(config, bit)

    for epoch in range(start_epoch, config["epoch"]+1):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["net_print"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u ,cls_out = net(image)
            loss = criterion(u,cls_out,label.float(), ind, config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        summaryWriter.add_scalar("training_loss",train_loss, epoch)
        f.write('Train | Epoch: %d | Loss: %.3f\n' % (epoch, train_loss))

        # if (epoch) % config["test_map"] == 0:
            # print("calculating test binary code......")
        tst_binary, tst_label = compute_result(test_loader, net, device=device)

        # print("calculating dataset binary code.......")\
        trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

        # print("calculating map.......")
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                            config["topK"], summaryWriter, epoch)
        
        print("map",mAP)
        if mAP > Best_mAP:
            Best_mAP = mAP
            P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
            print(f'Precision Recall Curve data:\n"DSH":[{P},{R}],')
            f.write('PR | Epoch %d | ' % (epoch))
            for PR in range(len(P)):
                f.write('%.5f %.5f ' % (P[PR], R[PR]))
            f.write('\n')
        
            print("Saving in ", config["save_path"])
            state = {
                'net': net.state_dict(),
                'Best_mAP': Best_mAP,
                'epoch': epoch,
            }
            torch.save(state, best_path)
        print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
            config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
        f.write('Test | Epoch %d | MAP: %.3f | Best MAP: %.3f\n'
            % (epoch, mAP, Best_mAP))
        print(config)

        state = {
            'net': net.state_dict(),
            'Best_mAP': Best_mAP,
            'epoch': epoch,
        }
        torch.save(state, trained_path)
        '''state = {
            'net': net.state_dict(),
            'Best_mAP': Best_mAP,
            'epoch': epoch,
        }
        torch.save(state, trained_path)'''
    f.close()


class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, class_out, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()

        loss_Cross = torch.nn.CrossEntropyLoss()
        # one-hot to labels
        lables = torch.argmax(y,dim=1)
        loss_c = loss_Cross(class_out,lables)  # 分类的交叉熵损失

        return center_loss + config["lambda"] * Q_loss + loss_c

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center
    
    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets

# class CSQLoss(torch.nn.Module):
#     def __init__(self, config, bit):
#         super(CSQLoss, self).__init__()
#         self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
#         self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
#         self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
#         self.criterion = torch.nn.BCELoss().to(config["device"])

#     def forward(self, u, y, ind, config):
#         u = u.tanh()
#         hash_center = self.label2center(y)
#         center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

#         Q_loss = (u.abs() - 1).pow(2).mean()
#         return center_loss + config["lambda"] * Q_loss

#     def label2center(self, y):
#         if self.is_single_label:
#             hash_center = self.hash_targets[y.argmax(axis=1)]
#         else:
#             # to get sign no need to use mean, use sum here
#             center_sum = y @ self.hash_targets
#             random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
#             center_sum[center_sum == 0] = random_center[center_sum == 0]
#             hash_center = 2 * (center_sum > 0).float() - 1
#         return hash_center

#     # use algorithm 1 to generate hash centers
#     def get_hash_targets(self, n_class, bit):
#         H_K = hadamard(bit)
#         H_2K = np.concatenate((H_K, -H_K), 0)
#         hash_targets = torch.from_numpy(H_2K[:n_class]).float()

#         if H_2K.shape[0] < n_class:
#             hash_targets.resize_(n_class, bit)
#             for k in range(20):
#                 for index in range(H_2K.shape[0], n_class):
#                     ones = torch.ones(bit)
#                     # Bernouli distribution
#                     sa = random.sample(list(range(bit)), bit // 2)
#                     ones[sa] = -1
#                     hash_targets[index] = ones
#                 # to find average/min  pairwise distance
#                 c = []
#                 for i in range(n_class):
#                     for j in range(n_class):
#                         if i < j:
#                             TF = sum(hash_targets[i] != hash_targets[j])
#                             c.append(TF)
#                 c = np.array(c)

#                 # choose min(c) in the range of K/4 to K/3
#                 # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
#                 # but it is hard when bit is  small
#                 if c.min() > bit / 4 and c.mean() >= bit / 2:
#                     print(c.min(), c.mean())
#                     break
#         return hash_targets


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)

