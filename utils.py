import numpy as np
import torch
import random
import imutils
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
import matplotlib.pyplot as plt

ohe = OneHotEncoder()
ohe.fit([[0], [1]])


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train_run_cls(model, train_loader, optimizer, criterion, epoch, N_CLASSES=2):
    # Set training mode
    model.train()
    trn_loss = AverageMeter()
    gt_train_onehot = []
    score_train_onehot = []
    for batch_idx, (inputs, _, targets) in enumerate(train_loader):

        inputs = Variable(inputs.cuda()).unsqueeze(1).float()
        targets = targets.to(device='cuda', dtype=torch.int64).squeeze()

        optimizer.zero_grad()  

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        trn_loss.update(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx + 1, loss.item()))

        # convert target, output to onehot probability array
        score = torch.softmax(outputs, dim=1)
        gt_onehot = ohe.transform(targets.cpu().reshape(-1, 1)).toarray()
        if batch_idx == 0:
            gt_train_onehot = gt_onehot
            score_train_onehot = score
        else:
            gt_train_onehot = np.concatenate((gt_train_onehot, gt_onehot), axis=0)
            score_train_onehot = torch.cat((score_train_onehot, score), axis=0)

    # compute train acc, auc
    y_gt_ = np.argmax(gt_train_onehot, 1)
    _, y_pred_ = torch.max(score_train_onehot, 1)
    train_acc = accuracy_score(y_gt_, y_pred_.cpu())
    score_train_onehot = score_train_onehot.detach().cpu()
    auc_all = []
    for i in range(N_CLASSES):
        auc_all.append(roc_auc_score(gt_train_onehot[:, i], score_train_onehot[:, i]))
    return trn_loss.avg, train_acc, auc_all


def val_run_cls(model, val_loader, criterion, N_CLASSES=2):
    losses = AverageMeter()
    gt_allval_onehot = []
    score_allval_onehot = []
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(val_loader):
            # print(flnm)
            inputs = Variable(inputs.cuda()).unsqueeze(1).float()
            targets = targets.to(device='cuda', dtype=torch.int64).squeeze()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item())

            # convert target, output to onehot probability array
            score = torch.softmax(outputs, dim=1)

            gt_onehot = ohe.transform(targets.cpu().reshape(-1, 1)).toarray()
            if batch_idx == 0:
                gt_allval_onehot = gt_onehot
                score_allval_onehot = score
            else:
                gt_allval_onehot = np.concatenate((gt_allval_onehot, gt_onehot), axis=0)
                score_allval_onehot = torch.cat((score_allval_onehot, score), axis=0)
    # compute val acc, auc
    y_gt_ = np.argmax(gt_allval_onehot, 1)
    _, y_pred_ = torch.max(score_allval_onehot, 1)
    acc = accuracy_score(y_gt_, y_pred_.cpu())
    score_allval_onehot = score_allval_onehot.cpu()
    auc_all = []
    for i in range(N_CLASSES):
        auc_all.append(roc_auc_score(gt_allval_onehot[:, i], score_allval_onehot[:, i]))

    return losses.avg, acc, auc_all


class Rotate:
    '''
    rotate img
    angle: [-15, 15]
    '''

    def __init__(self):
        self.angle = random.randint(-10, 10)
        self.prob = random.random()

    def __call__(self, data):
        if self.prob > 0.5:
            data = imutils.rotate(data, self.angle)
        return data


class Translate:
    '''
    translate img
    x axis: [-15, 15]
    y axis: [-15, 15]
    '''

    def __init__(self):
        self.x = random.randint(-10, 10)
        self.y = random.randint(-10, 10)
        self.prob = random.random()

    def __call__(self, data):
        if self.prob > 0.5:
            data = imutils.translate(data, self.x, self.y)
        return data


class Flip:
    '''
    translate img
    x axis: [-15, 15]
    y axis: [-15, 15]
    '''

    def __init__(self):
        self.prob = random.random()

    def __call__(self, data):
        if self.prob < 0.5:
            data = np.fliplr(data)
        return data
