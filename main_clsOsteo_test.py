from readData import ListDatasetFloatMat_cw, ListDatasetFloatMat_zw
import torch
from torch.autograd import Variable
import numpy as np
import os
from networks.resNet18 import ResNet18
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import accuracy_score, precision_recall_curve
import scipy.io as sio


ohe = OneHotEncoder()
ohe.fit([[0], [1]])


# paras
use_cw = True
device = 'cuda'
N_CLASSES = 2
weight_path = './ckpts'
model = ResNet18(N_CLASSES).to(device)


# read test data
test_dset = ListDatasetFloatMat_cw(data_folder=r'./Data', list_file='./datalist/test_subject.txt')
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, num_workers=4, shuffle=False, drop_last=False)


# load model
if use_cw:
    weights_fname = 'weights-cw.pth'
else:
    weights_fname = 'weights-zw.pth'

weights_fpath = os.path.join(weight_path, weights_fname)
model.load_state_dict(torch.load(weights_fpath)['state_dict'])

model.eval()
with torch.no_grad():
    for batch_idx, (inputs, _, targets) in enumerate(test_loader):
        inputs = Variable(inputs.to(device)).unsqueeze(1).float()
        targets = targets.to(device=device, dtype=torch.int64).squeeze()

        outputs = model(inputs)
        score = torch.softmax(outputs, dim=1)

        gt_onehot = ohe.transform(targets.cpu().reshape(-1, 1)).toarray()
        if batch_idx == 0:
            gt_test_onehot = gt_onehot
            score_test_onehot = score
        else:
            gt_train_onehot = np.concatenate((gt_test_onehot, gt_onehot), axis=0)
            score_test_onehot = torch.cat((score_test_onehot, score), axis=0)
    # compute train acc, auc
    y_gt_ = np.argmax(gt_test_onehot, 1)
    _, y_pred_ = torch.max(score_test_onehot, 1)

    test_acc = accuracy_score(y_gt_, y_pred_.cpu())
    score_test_onehot = score_test_onehot.detach().cpu()
    auc_all = []
    for i in range(N_CLASSES):
        auc_all.append(roc_auc_score(gt_test_onehot[:, i], score_test_onehot[:, i]))

    print('epoch: %d, acc: %f, auc_os: %f, auc_no:%f' % (epoch, test_acc, auc_all[0], auc_all[1]))


