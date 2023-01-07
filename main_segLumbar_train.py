import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from readData import ListData4Seg
from tensorboardX import SummaryWriter
from networks.Unet import U_Net
from utils import AverageMeter, Rotate, Translate
from dice_loss import dice_coefficient
import matplotlib.pyplot as plt


# paras
device = 'cuda'
use_val = False
use_cw = True
RUNS = 30
Summary_DIR = './summarys/UNet_seg'
weight_path = './ckpts/UNet_seg'
start_epoch = 0


# path
if use_cw:
    weight_path = './ckpts/UNet_seg' + '_cw'
    Summary_DIR = './summarys/UNet_seg'+ '_cw'

if not os.path.exists(Summary_DIR):
    os.makedirs(Summary_DIR)

if not os.path.exists(weight_path):
    os.makedirs(weight_path)


# Data pre-processing
transforms_train = [Rotate(),
                    Translate()]

# load data
train_dset = ListData4Seg(data_folder=r'./Data', list_file='./datalist/train_subject.txt', use_cw=use_cw,
                          transform=None)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=4, num_workers=4, shuffle=True, drop_last=use_cw)

val_dset = ListData4Seg(data_folder=r'./Data', list_file='./datalist/val_subject.txt', use_cw=True,
                        transform=None)
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=8, num_workers=4, shuffle=False, drop_last=True)


model = U_Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# summarywriter
train_writer = SummaryWriter(os.path.join(Summary_DIR, 'train'), flush_secs=2)

if use_val:
    val_writer = SummaryWriter(os.path.join(Summary_DIR, 'val'), flush_secs=2)

# load model
if start_epoch:
    weights_fname = 'weights-%d.pth' % start_epoch
    weights_fpath = os.path.join(weight_path, weights_fname)
    model.load_state_dict(torch.load(weights_fpath)['state_dict'])

cnt = 1
for epoch in range(start_epoch, RUNS):

    trn_loss = AverageMeter()
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        inputs = Variable(inputs.cuda()).unsqueeze(1).float()
        targets = Variable(targets.cuda()).unsqueeze(1).float()

        optimizer.zero_grad()  

        outputs = model(inputs)
        # print(outputs.min(), outputs.max())

        loss = dice_coefficient(outputs, targets)
        trn_loss.update(loss.item())
        loss.backward()
        optimizer.step()

        train_writer.add_scalar('model/tlossStep', loss, cnt)
        cnt = cnt + 1
        if cnt % 10 == 0:
            print('step %d:  train_loss: %f' % (cnt, loss))

    train_writer.add_scalar('model/tlossEpoch', trn_loss.avg, epoch)
    print('Epoch %d:  train_loss: %f' % (epoch, trn_loss.avg))

    # checkpoint path
    weights_fname = 'weights-%d.pth' % epoch
    weights_fpath = os.path.join(weight_path, weights_fname)

    if use_val:
        model.eval()
        with torch.no_grad():
            val_loss = AverageMeter()
            for batch_idx, (inputs, targets, _) in enumerate(train_loader):
                inputs = inputs.to(device).unsqueeze(1).float()
                targets = targets.to(device).unsqueeze(1).float()
                outputs = model(inputs)
                vloss = dice_coefficient(outputs, targets)
                val_loss.update(vloss.item())

            val_writer.add_scalar('model/vlossEpoch', val_loss.avg, epoch)

            print('Epoch %d:  val_loss: %f' % (epoch, val_loss.avg))

    torch.save({
        'train_loss': trn_loss.avg,
        'startEpoch': epoch,
        'state_dict': model.state_dict()}, weights_fpath)
