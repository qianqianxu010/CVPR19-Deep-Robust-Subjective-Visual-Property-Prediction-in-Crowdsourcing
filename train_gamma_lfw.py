from __future__ import print_function, absolute_import, division

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import LFWDataSet, LFWImgDataSet, OutlierLossL2, OutlierLossLogistic
from torchvision import transforms
from torchvision.models import resnet50, vgg16_bn, alexnet
import time
import pdb
import os
import math
import argparse

# pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--max_epoch', type=int, default=50,
                    help='#epochs to update network params successively')
parser.add_argument('--gamma_epoch', type=int, default=10,
                    help='#epochs to update gamma successively')
parser.add_argument('-m', '--model', type=str, default='resnet',
                    choices=['resnet', 'vgg', 'alexnet'], help='model name')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-L', type=float, default=10)
parser.add_argument('--lamda', type=float, default=1.2,
                    help='regularization param for gamma')
parser.add_argument('--outer', type=int, default=3,
                    help='#outer_epochs to alternatively update network params and gamma')
parser.add_argument('--train_file', type=str, default='data/train_LFW10.txt',
                    help='training set filename')
parser.add_argument('--dataset', type=str, default='lfw', help='model_name')
parser.add_argument('--test_file', type=str, default='data/test_LFW10.txt',
                    help='test set filename')
parser.add_argument('--loss', type=str, default='l2',
                    help='loss type l2 for linear, logit for logistic regression',
                    choices=['l2', 'logit'])
parser.add_argument('--attrib_num', type=int, default=10,
                    help='# of attributes')
parser.add_argument('--binary_label', type=bool, default=False,
                    help='using binary label (0/1) or not (-1/1)')
args = parser.parse_args()

DIRFILE = os.path.join(
    '.', '{}_{}_lambda_{:.1f}'.format(args.dataset, args.loss, args.lamda))
WRITTERPATH = os.path.join(DIRFILE, 'with_gamma_{:}'.format(args.loss))
print('lambda :{:2.4f}'.format(args.lamda))
all_pair_num, pairs_per_attr = 14587, 500


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# ================= load pretrained backbone =====================
print('Select pretrained model--', args.model)

if args.model == 'resnet':
    model = resnet50(pretrained=True)
    model.fc = Identity()
    model = model.cuda()
    model_dim = 2048
    train_batch_size = 32
    test_batch_size = 32
elif args.model == 'alexnet':
    model = alexnet(pretrained=False).cuda()
    dicts = torch.load('./face_alexnet_checkpt_80.pkl')
    model_state_dict = dicts['model_state_dict']
    model.load_state_dict(model_state_dict)
    model_dim = 4096
    train_batch_size = 128
    test_batch_size = 128
elif args.model == 'vgg':
    model = vgg16_bn(pretrained=True)
    model.classifier._modules['6'] = Identity()
    model = model.cuda()
    model_dim = 4096
    train_batch_size = 32
    test_batch_size = 32
else:
    print('please select proper model such as \'alexnet\'')
    quit()


# ================= load dataset =====================
kwargs = {'num_workers': 8, 'pin_memory': True}
train_transform = transforms.Compose(
    # [transforms.Resize((256, 256)),
    #  transforms.RandomCrop((224, 224)),
    #  transforms.RandomHorizontalFlip(),
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])

# pairwise comparison: (i, j, label, attr_id, pair_id, strength)
# e.g.: for attribute 1, there are 5 comparisons between 3rd pair(i, j):
#       (i, j, 1) * 3, (i, j, -1) * 2.  Then the samples will be:
#       (i, j, 1, 1, 3, 3) * 1, (i, j, -1, 1, 3, 2) * 1
train_pair_data_with_img = LFWDataSet(txt=args.train_file,
                                      binary=args.binary_label, load_img=True,
                                      transform=train_transform)
train_loader = DataLoader(dataset=train_pair_data_with_img, batch_size=train_batch_size,
                          shuffle=True, **kwargs)
print('loading train_pair_data:', len(train_pair_data_with_img))

train_pair_data_wo_img = LFWDataSet(txt=args.train_file,
                                    binary=args.binary_label, load_img=False)
train_loader_wo_img = DataLoader(dataset=train_pair_data_wo_img, batch_size=train_batch_size,
                                 shuffle=True, **kwargs)

# single image: (img, img_id, attr_id)
train_img_data = LFWImgDataSet(txt=args.train_file,
                               transform=train_transform)
train_img_loader = DataLoader(dataset=train_img_data, batch_size=train_batch_size,
                              shuffle=False, **kwargs)
print('loading train_img_data:', len(train_img_data))

test_data = LFWDataSet(txt=args.test_file,
                       binary=args.binary_label,
                       transform=train_transform)
test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size,
                         shuffle=False, **kwargs)
print('loading test_data:', len(test_data))

print('Traing sample number: {}'.format(all_pair_num))
print('Traing outlier ratio: {:.4f}'.format(
    train_pair_data_with_img.get_outlier_num() / all_pair_num))

# ============= linear classifiers for global features ================
score_functions = nn.ModuleList(
    [nn.Linear(model_dim, 1).cuda() for i in range(args.attrib_num)])


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, 0.01)
        nn.init.constant_(m.bias.data, 0)


for i in range(args.attrib_num):
    weight_init(score_functions[i])

# ================= loss function =====================
gammas = torch.zeros((args.attrib_num, pairs_per_attr, 2)).cuda()
gammas_n = torch.zeros((args.attrib_num, pairs_per_attr, 2)).cuda()
gammas_old = torch.zeros((args.attrib_num, pairs_per_attr, 2)).cuda()
strengths = torch.zeros((args.attrib_num, pairs_per_attr, 2))
zero = torch.zeros((1)).cuda()
lamda = args.lamda
if args.loss == 'l2':
    loss_fn = OutlierLossL2(lamda=lamda)
elif args.loss == 'logit':
    loss_fn = OutlierLossLogistic(lamda=lamda, binary=args.binary_label)
else:
    print('loss should either be l2 or logit, please choose again!')
    quit()
print('ranking loss:', loss_fn)

# ================= hyper-parameters =====================
lr = args.learning_rate
print('Initial lr:', lr)
params_a = [{'params': model.parameters()}, {
    'params': score_functions.parameters()}]
# update one part params.
# optimizer_a = optim.SGD(params_a, lr=lr, momentum=0.9)
optimizer_a = optim.Adam(params_a, lr=lr)
multisteps = [20, 40, 60, 90, 120]
scheduler_a = optim.lr_scheduler.MultiStepLR(
    optimizer_a, milestones=multisteps, gamma=0.1)
# use CUDA
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
# show interval
log_interval = 20
SAVE_INTERVAL = 5


if not os.path.exists(DIRFILE):
    os.mkdir(DIRFILE)


def train(epoch):
    model.train()
    score_functions.train()

    loss = 0
    batch_cnt = 0
    for step, (_, _, img1, img2, label, attr_id, pair_id, strength) in enumerate(train_loader):
        img1, img2, label, attr_id, pair_id, strength = img1.cuda(), img2.cuda(
        ), label.cuda(), attr_id.cuda(), pair_id.cuda(), strength.cuda()
        optimizer_a.zero_grad()
        bz = label.shape[0]
        output1 = model(img1)
        output2 = model(img2)
        train_score1 = torch.zeros(bz).cuda()
        train_score2 = torch.zeros(bz).cuda()
        gamma = torch.zeros(bz).cuda()
        for k in range(bz):  # for a batch
            idx = int(attr_id[k])
            train_score1[k] = score_functions[idx](output1[k])
            train_score2[k] = score_functions[idx](output2[k])
            gamma[k] = gammas[idx, int(pair_id[k]), int((label[k] + 1) / 2)]
            strengths[idx, int(pair_id[k]), int(
                (label[k] + 1) / 2)] = strength[k]
        cls_loss = loss_fn(train_score1, train_score2, label, gamma, strength)
        cls_loss.backward()
        optimizer_a.step()

        loss += cls_loss.item()
        batch_cnt += 1

        # show info.
        # if step % log_interval == 0:
        #     print('Time:', time.asctime(time.localtime(time.time())),
        #           '|Epoch:', epoch,
        #           '|cls_loss:', cls_loss.item()
        #           )
    print('Time:', time.asctime(time.localtime(time.time())),
          '|Epoch:', epoch,
          '|total_loss:', loss / batch_cnt
          )
    if epoch - 1 in multisteps:
        print('adjust learning rate...')
        for param_group in optimizer_a.param_groups:
            print('lr:', param_group['lr'])

    # save params
    if epoch % SAVE_INTERVAL == 0:
        cptname = 'naive_{:}_checkpt_{:}.pkl'.format(args.dataset, epoch)
        cptpath = os.path.join(DIRFILE, cptname)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'score_funcs_dict': score_functions.state_dict(),
                    'optimizer': optimizer_a.state_dict()},
                   cptpath)

    return loss


def test(epoch, training_set=False, dataloader=test_loader):
    model.eval()
    score_functions.eval()
    test_loss = 0
    correct = 0
    batch_cnt = 0
    fw = open(WRITTERPATH, 'w')
    with torch.no_grad():
        for step, (_, _, img1, img2, label, attr_id, pair_id, strength) in enumerate(dataloader):
            img1, img2, label, attr_id, pair_id, strength = img1.to(device), img2.to(
                device), label.to(device), attr_id.to(device), pair_id.to(device), strength.to(device)
            output1 = model(img1)
            output2 = model(img2)
            bz = attr_id.shape[0]
            test_score1 = torch.zeros(bz).cuda()
            test_score2 = torch.zeros(bz).cuda()
            gamma = torch.zeros(bz).cuda()
            for k in range(bz):  # for a batch
                idx = int(attr_id[k])
                test_score1[k] = score_functions[idx](output1[k])
                test_score2[k] = score_functions[idx](output2[k])
                if training_set:
                    gamma[k] = gammas[idx, int(
                        pair_id[k]), int((label[k] + 1) / 2)]
            if training_set:
                cls_loss = loss_fn(test_score1, test_score2,
                                   label, gamma, strength)
                test_loss += cls_loss.item()
            batch_cnt += 1
            pred = test_score1 - test_score2
            for k in range(bz):
                fw.write('Attribute: %d, Pair: %d, Label: %d, Score: %.4f, Strength: %d\n' %
                         (attr_id[k], pair_id[k], label[k], pred[k], strength[k]))
            pred[pred > 0] = 1
            pred[pred <= 0] = 0 if args.binary_label else -1
            pred = pred.type(torch.cuda.LongTensor)
            label = label.type(pred.dtype)
            correct += pred.eq(label.view_as(pred)).sum().item()
    fw.close()

    if training_set:
        test_loss /= batch_cnt
        print('Training set {}: average loss:{:.4f}, acc:{:.4f}'.format(
            len(dataloader.dataset), test_loss, correct / len(dataloader.dataset)))
    else:
        print('Test set {}: acc:{:.4f}'.format(
            len(dataloader.dataset), correct / len(dataloader.dataset)))


def logit_fun(s, gamma):
    return 1 / (1 + torch.exp(-(s + gamma)))


def solve_l1(x, lamd):
    return torch.max(torch.abs(x) - lamd, zero) * torch.sign(x)


# inference the scores for all the images
def get_all_scores(dataloader=train_img_loader):
    dict_scores = dict()
    with torch.no_grad():
        for step, (img, img_id, attr_id) in enumerate(dataloader):
            img_id, img, attr_id = img_id.to(
                device), img.to(device), attr_id.to(device)
            output = model(img)
            bz = attr_id.shape[0]
            for k in range(bz):
                idx = int(attr_id[k])
                dict_scores[(int(img_id[k]), idx)] = score_functions[idx](
                    output[k]).item()
    return dict_scores


def update_gammas(type, dataloader=train_loader_wo_img):
    model.eval()
    score_functions.eval()
    dict_scores = get_all_scores()
    if type == 'l2':
        with torch.no_grad():
            for step, (img_id1, img_id2, label, attr_id, pair_id, strength) in enumerate(dataloader):
                label, attr_id, pair_id = label.to(
                    device), attr_id.to(device), pair_id.to(device)
                bz = attr_id.shape[0]
                for k in range(bz):  # for a batch
                    idx = int(attr_id[k])
                    test_score1 = dict_scores[(int(img_id1[k]), idx)]
                    test_score2 = dict_scores[(int(img_id2[k]), idx)]
                    delta = label[k] - (test_score1 - test_score2)
                    gammas[idx, int(pair_id[k]), int(
                        (label[k] + 1) / 2)] = solve_l1(delta, lamda)
    elif type == 'logit':
        L = args.L
        t_new = 1
        with torch.no_grad():
            for _ in range(args.gamma_epoch):
                gammas_old = gammas
                t_old = t_new
                t_new = (1 + math.sqrt(1 + 4 * t_old * t_old)) / 2
                dt = (t_old - 1) / t_new
                for step, (img_id1, img_id2, label, attr_id, pair_id, strength) in enumerate(dataloader):
                    label, attr_id, pair_id, strength = label.to(device), attr_id.to(
                        device), pair_id.to(device), strength.to(device)
                    bz = label.shape[0]
                    for k in range(bz):  # for a batch
                        idx = int(attr_id[k])
                        pair_idx = int(pair_id[k])
                        label_k = int(label[k]) if args.binary_label else int(
                            (label[k] + 1) / 2)
                        test_score1 = dict_scores[(int(img_id1[k]), idx)]
                        test_score2 = dict_scores[(int(img_id2[k]), idx)]
                        gamma = gammas_n[idx, pair_idx, label_k]
                        h = logit_fun(test_score1 - test_score2, gamma)
                        delta = gamma - (1 / L) * strength[k] * (h - label[k])
                        zeta = strength[k] * lamda / L
                        gammas[idx, pair_idx, label_k] = solve_l1(delta, zeta)
                        gammas_n[idx, pair_idx, label_k] = gammas[idx, pair_idx, label_k] + \
                            dt * (gammas[idx, pair_idx, label_k] -
                                  gammas_old[idx, pair_idx, label_k])

    else:
        print('loss could either be l2 or logit')
        quit()


# ===================== training/testing =================
EPOCHS = args.max_epoch
OUTERS = args.outer
for outer in range(OUTERS):
    last_loss = -1
    for epoch in range(1, EPOCHS + 1):
        scheduler_a.step()
        train_loss = train(epoch)
        if last_loss > -1 and abs(train_loss - last_loss) < 1e-4:
            print('Early stopping...')
            test(epoch)
            break
        last_loss = train_loss
    print('Training loss before update gamma...')
    test(epoch, True, train_loader)
    print('Update gamma...')
    update_gammas(args.loss)
    print('Training loss after update gamma...')
    test(epoch, True, train_loader)
    test(epoch)

cnt_outlier = 0
threshold = 1e-6 if args.loss is 'l2' else 1e-5
OUTLIERPATH = os.path.join(DIRFILE, 'outliers.txt')
GAMMAPATH = os.path.join(DIRFILE, 'gamma.mat')
with open(OUTLIERPATH, 'w') as f:
    for i in range(args.attrib_num):
        for j in range(pairs_per_attr):
            if abs(gammas[i, j, 0].item()) > threshold:
                cnt_outlier += strengths[i, j, 0].item()
                f.write('Attribute: %d, Pair: %d, Label: %d, Gamma: %.4f\n' %
                        (i, j, -1, gammas[i, j, 0]))
            if abs(gammas[i, j, 1].item()) > threshold:
                cnt_outlier += strengths[i, j, 1].item()
                f.write('Attribute: %d, Pair: %d, Label: %d, Gamma: %.4f\n' %
                        (i, j, 1, gammas[i, j, 1]))
print('Training outlier ratio by the model: {:.4f}%'.format(
    100 * cnt_outlier / all_pair_num))

sio.savemat(GAMMAPATH, {'gammas': gammas.cpu().numpy()})
