import os
import torch
import torch.nn as nn
import torch.optim as optim
from readData import ListDatasetFloatMat_cw, ListDatasetFloatMat_zw
from tensorboardX import SummaryWriter
from networks.resNet18 import ResNet18
from utils import train_run_cls, val_run_cls, Rotate, Translate, Flip

# Data pre-processing
transforms_train = [Flip(), Translate()]

# load data
train_dset = ListDatasetFloatMat_zw(data_folder=r'./Data', list_file='./datalist/train_subject.txt',
                                    transform=transforms_train)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=16, num_workers=4, shuffle=True, drop_last=True)

val_dset = ListDatasetFloatMat_zw(data_folder=r'./Data', list_file='./datalist/val_subject.txt',
                                  transform=None)
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=8, num_workers=4, shuffle=False, drop_last=True)

# paras
start_epoch = 100
device = 'cuda'
use_val = True
N_CLASSES = 2
RUNS = 150
Summary_DIR = './summarys'
weight_path = './ckpts'
model = ResNet18(N_CLASSES).to(device)

# path
if not os.path.exists(Summary_DIR):
    os.makedirs(Summary_DIR)

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

# loss
criterion = nn.CrossEntropyLoss().cuda()

# summarywriter
train_writer = SummaryWriter(os.path.join(Summary_DIR, 'train'), flush_secs=2)

if use_val:
    val_writer = SummaryWriter(os.path.join(Summary_DIR, 'val'), flush_secs=2)

#load model
weights_fname = 'weights-%d.pth' % (start_epoch - 1)
weights_fpath = os.path.join(weight_path, weights_fname)
model.load_state_dict(torch.load(weights_fpath)['state_dict'])

for epoch in range(start_epoch, RUNS):
    # lr and optimizer
    lr = 0.01
    if epoch >= 50: lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ## Train ###
    trn_loss, trn_acc, trn_auc = train_run_cls(model, train_loader, optimizer, criterion, epoch)
    # writer
    train_writer.add_scalar('loss', trn_loss, epoch)
    train_writer.add_scalar('auc_os', trn_auc[0], epoch)
    train_writer.add_scalar('auc_no', trn_auc[1], epoch)
    train_writer.add_scalar('acc', trn_acc, epoch)
    print('Epoch %d:  train_loss: %f' % (epoch, trn_loss))
    print('Epoch for train%d:  acc: %f, auc_os: %f, auc_no:%f' % (epoch, trn_acc, trn_auc[0], trn_auc[1]))

    # checkpoint path
    weights_fname = 'weights-%d.pth' % epoch
    weights_fpath = os.path.join(weight_path, weights_fname)

    if use_val:
        val_loss, val_acc, val_auc = val_run_cls(model, val_loader, criterion)
        val_writer.add_scalar('loss', val_loss, epoch)
        val_writer.add_scalar('auc_os', val_auc[0], epoch)
        val_writer.add_scalar('auc_no', val_auc[1], epoch)
        val_writer.add_scalar('acc', val_acc, epoch)
        print('Epoch %d:  val_loss: %f' % (epoch, val_loss))
        print('Epoch for val%d:  acc: %f, auc_os: %f, auc_no:%f' % (epoch, val_acc, val_auc[0], val_auc[1]))

    torch.save({
        'train_loss': trn_loss,
        'startEpoch': epoch,
        'state_dict': model.state_dict()}, weights_fpath)

train_writer.close()
if use_val:
    val_writer.close()
