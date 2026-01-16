import os
import torch
from torch import nn
import random
import numpy as np
import matplotlib
from utils.load_data import load_PACS, load_officehome
from utils.ResNet18 import ResNet_Client, ResNet_Server, BasicBlock
matplotlib.use('Agg')
import copy
import argparse

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

def prRed(skk): print("\033[91m {}\033[00m".format(skk))
def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []
run_time = []
global_acc_test = []
global_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc

acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []


idx_collect = []
l_epoch_check = False
fed_check = False


gra_list_for_client = []
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, global_iter):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user, lr
    global gra_list_for_client
    global gra_list, gra_common

    net_glob_server.train()
    optimizer_server = torch.optim.SGD(net_glob_server.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    optimizer_server.zero_grad()
    fx_client = fx_client.to(device)
    fx_client.requires_grad_()
    y = y.to(device)
    # ---------forward prop-------------
    fx_server = net_glob_server(fx_client)

    loss_ce = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------

    loss_ce.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    #------------update mask-----------------------
    dfx_client_abs = torch.abs(dfx_client)
    mask = generate_mask_from_gradient(dfx_client_abs, gra_common, drop_rate, global_iter)
    gra_list_for_client.append(dfx_client_abs)


    batch_loss_train.append(loss_ce.item())
    batch_acc_train.append(acc.item())


    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        avg_gra_for_client = sum(gra_list_for_client) / len(gra_list_for_client)
        gra_list[idx] = avg_gra_for_client
        gra_list_for_client = []

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))
        if l_epoch_count == l_epoch - 1:

            l_epoch_check = True

            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            if idx not in idx_collect:
                idx_collect.append(idx)

        if len(idx_collect) == num_users:
            fed_check = True


            gra_common = sum(gra_list)/len(gra_list)

            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    return dfx_client, mask


def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net_glob_server.eval()

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        # ---------forward prop-------------
        fx_server = net_glob_server(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                                             loss_avg_test))

            if l_epoch_check:
                l_epoch_check = False

                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            if fed_check:
                fed_check = False

                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER ==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                         loss_avg_all_user))
                print("==========================================================")

    return


def glob_eval():
    global net_glob_server, test_loader, criterion, device, loss_test_collect, acc_test_collect
    net_server = copy.deepcopy(net_glob_server).to(device)
    net_server.eval()
    net_client = copy.deepcopy(net_glob_client).to(device)
    net_client.eval()
    loss_test_global = []
    acc_test_global = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            #---------forward prop-------------
            fx_client = net_client(images)
            fx_server = net_server(fx_client)

            # calculate loss
            loss = criterion(fx_server, labels)
            # calculate accuracy
            acc = calculate_accuracy(fx_server, labels)

            loss_test_global.append(loss.item())
            acc_test_global.append(acc.item())

        acc_avg_test = sum(acc_test_global)/len(acc_test_global)
        loss_avg_test = sum(loss_test_global)/len(loss_test_global)
        global_loss_test.append(loss_avg_test)
        global_acc_test.append(acc_avg_test)
        print("==========================================================")
        print('Global Test: Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(acc_avg_test, loss_avg_test))
        print("==========================================================")


loss_test_collect_for_domains = []
acc_test_collect_for_domains = []
def glob_eval_for_domain():
    global net_glob_server, test_loaders_for_domains_list, test_indices_for_domains, criterion, device, loss_test_collect_for_domains, acc_test_collect_for_domains, test_loader
    net_server = copy.deepcopy(net_glob_server).to(device)
    net_server.eval()
    net_client = copy.deepcopy(net_glob_client).to(device)
    net_client.eval()
    n_domain = len(test_loaders_for_domains)
    loss_collect_for_all = []
    acc_collect_for_all = []
    for i in range(n_domain):
        loss_test_domain = []
        acc_test_domain = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loaders_for_domains_list[i]):
                images, labels = images.to(device), labels.to(device)
                #---------forward prop-------------
                fx_client = net_client(images)
                fx_server = net_server(fx_client)

                # calculate loss
                loss = criterion(fx_server, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx_server, labels)

                loss_test_domain.append(loss.item())
                acc_test_domain.append(acc.item())

            acc_avg_test = sum(acc_test_domain)/len(acc_test_domain)
            loss_avg_test = sum(loss_test_domain)/len(loss_test_domain)
        loss_collect_for_all.append(loss_avg_test)
        acc_collect_for_all.append(acc_avg_test)

    loss_test_con_domain = []
    acc_test_con_domain = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            # ---------forward prop-------------
            fx_client = net_client(images)
            fx_server = net_server(fx_client)

            # calculate loss
            loss = criterion(fx_server, labels)
            # calculate accuracy
            acc = calculate_accuracy(fx_server, labels)

            loss_test_con_domain.append(loss.item())
            acc_test_con_domain.append(acc.item())

        acc_avg_test = sum(acc_test_con_domain) / len(acc_test_con_domain)
        loss_avg_test = sum(loss_test_con_domain) / len(loss_test_con_domain)

        loss_collect_for_all.append(loss_avg_test)
        acc_collect_for_all.append(acc_avg_test)

    loss_test_collect_for_domains.append(loss_collect_for_all)
    acc_test_collect_for_domains.append(acc_collect_for_all)


def Init_mask(net_client_model, ldr_train):
    for (images, labels) in ldr_train:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            fx = net_client_model(images)
            mask = torch.rand_like(fx) > -999999  # 创建与 representation 相同形状的随机张量
            mask = mask.to(device)
            return mask


def mask_representation(representation, mask):
    if representation.size() != mask.size():
        mask = mask[:representation.size(0), :representation.size(1)]
    dropped_representation = representation * mask

    dropped_representation = dropped_representation.detach()
    dropped_representation.requires_grad_()

    return dropped_representation



def generate_mask_from_gradient(adx_client, glo_com, rate, gloval_iter):
    alpha_c = alpha * (1 - gloval_iter/args.epochs)
    weighted_grad_magnitude = (1.0-alpha_c) * adx_client + alpha_c * glo_com

    threshold = torch.quantile(weighted_grad_magnitude, 1-rate)
    mask = weighted_grad_magnitude <= threshold

    return mask


class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None,
                 idxs_test=None, mask=None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 5
        # self.selected_clients = []
        self.ldr_train = dataset_train
        self.ldr_test = dataset_test
        self.mask = Init_mask(net_client_model, dataset_train)

    def train(self, net, global_iter):
        net.train()
        optimizer_client = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                # ---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                dropped_fx_client = mask_representation(client_fx, self.mask)


                dfx, mask = train_server(dropped_fx_client, labels, iter, self.local_ep, self.idx, len_batch, global_iter)
                self.mask = mask


                # --------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()

        return net.state_dict()

    def evaluate(self, net, ell):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(images)

                # Sending activations to server
                evaluate_server(fx, labels, self.idx, len_batch, ell)

        return

    # =====================================================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Model training and evaluation")
    parser.add_argument('-dr', type=float, default=0.4, help='drop_rate')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-bs', type=float, default=32, help='batch size')
    parser.add_argument('-alpha', type=float, default=0.2, help='init weights for global')
    parser.add_argument('-model', type=str, default="GoogleNet", help='ResNet18, GoogleNet')
    parser.add_argument('-dataset', type=str, default="OfficeHome", help='PACS, OfficeHome')
    parser.add_argument('-cut_layer', type=int, default=4, help='1,2,3,4')
    args = parser.parse_args()
    return args


def init_gra_for_list_global(n_user, net_client_model, ldr_train):
    for (images, labels) in ldr_train:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            fx = net_client_model(images)
            zeors_grd = torch.zeros_like(fx)
            gra_list = [zeors_grd for _ in range(n_user)]
        return gra_list, zeors_grd





if __name__ == "__main__":
    args = get_args()
    args.epochs = 5
    frac = 1
    lr = args.lr
    drop_rate = args.dr
    batch_size = args.bs
    alpha = args.alpha
    num_classes = 7 if args.dataset == "PACS" else 65

    gpu = 2
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    print(device)

    if args.model == "ResNet18":
        net_glob_client = ResNet_Client(BasicBlock, [2, 2, 2, 2], nf=64, cut_layer=args.cut_layer)
        net_glob_client.to(device)
        print(net_glob_client)
        net_glob_server = ResNet_Server(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes, nf=64, cut_layer=args.cut_layer)
        net_glob_server.to(device)
        print(net_glob_server)
        net_glob_client.train()
        w_glob_client = net_glob_client.state_dict()
    elif args.model == "GoogleNet":
        from utils.GoogleNet import GoogLeNet_Client, GoogLeNet_Server
        net_glob_client = GoogLeNet_Client(cut_layer=args.cut_layer, pretrained=True)
        net_glob_client.to(device)
        print(net_glob_client)
        net_glob_server = GoogLeNet_Server(num_classes=num_classes, cut_layer=args.cut_layer, pretrained=True)
        net_glob_server.to(device)
        print(net_glob_server)
        net_glob_client.train()
        w_glob_client = net_glob_client.state_dict()



    if args.dataset == "PACS":
        train_loaders, test_loader, dict_users_train, test_indices, test_loaders_for_domains, test_indices_for_domains = load_PACS(batch_size=batch_size)
    elif args.dataset == "OfficeHome":
        train_loaders, test_loader, test_loaders_for_domains = load_officehome(batch_size=batch_size)
    else:
        raise ValueError(f"❌ : incorrect dataset name:'{args.dataset}'")


    train_loader_list = list(train_loaders.values())
    test_loaders_for_domains_list = list(test_loaders_for_domains.values())
    num_users = len(train_loader_list)
    client_list = []
    for idx in range(num_users):
        local_client = Client(net_glob_client, idx, lr, device, dataset_train = train_loader_list[idx], dataset_test = test_loader)
        client_list.append(local_client)

    gra_list, gra_common = init_gra_for_list_global(n_user=num_users, net_client_model=net_glob_client, ldr_train= train_loader_list[0])


    for iter in range(args.epochs):
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        w_locals_client = []
        for idx in idxs_users:
            local = client_list[idx]
            # Training ------------------
            w_client = local.train(net=copy.deepcopy(net_glob_client).to(device), global_iter=iter)
            w_locals_client.append(copy.deepcopy(w_client))

            # Testing -------------------
            local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)

        # Ater serving all clients for its local epochs------------
        # Federation process at Client-Side------------------------
        print("------------------------------------------------------------")
        print("------ Fed Server: Federation process at Client-Side -------")
        print("------------------------------------------------------------")
        w_glob_client = FedAvg(w_locals_client)

        # Update client-side global model
        net_glob_client.load_state_dict(w_glob_client)
        glob_eval()

        if args.epochs - iter < 30:
            glob_eval_for_domain()


    print("Training and Evaluation completed!")

    # ===============================================================================
    # Save output data to .excel file (we use for comparision plots)
    round_process = [i for i in range(1, len(acc_train_collect) + 1)]
    print(loss_train_collect)
    print(loss_test_collect)
    print(acc_train_collect)
    print(acc_test_collect)
    print(global_loss_test)
    print(global_acc_test)
    print(loss_test_collect_for_domains)
    print(acc_test_collect_for_domains)

    output_dir = (
        f"./results/{args.dataset}/"
        f"{args.model}_SFL_DiMask_l{args.cut_layer}_bs{args.bs}_lr{args.lr}"
        f"_epoch_{args.epochs}.txt"
    )

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with open(output_dir, 'w') as f:
        f.write(f"Round Process: {round_process}\n")
        f.write(f"Train Loss: {loss_train_collect}\n")
        f.write(f"Test Loss: {loss_test_collect}\n")
        f.write(f"Train Accuracy: {acc_train_collect}\n")
        f.write(f"Test Accuracy: {acc_test_collect}\n")
        f.write(f"Global Test Loss: {global_loss_test}\n")
        f.write(f"Global Test Accuracy: {global_acc_test}\n")
        f.write(f'Domains Test Loss: {loss_test_collect_for_domains}\n')
        f.write(f'Domains Test Accuracy: {acc_test_collect_for_domains}\n')




# =============================================================================
#                         Program Completed
# =============================================================================








