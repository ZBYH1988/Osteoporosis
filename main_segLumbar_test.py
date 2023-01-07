import os
import numpy as np
import scipy.io as sio
import torch
from readData import ListData4Seg
from networks.Unet import U_Net
from dice_loss import dice_coefficient
import matplotlib.pyplot as plt


device = 'cuda'
outDim = 1
use_cw = True
weight_path = './ckpts/UNet_seg'
save_path = './results/UNet_seg'

model = U_Net().to(device)

# read test data
test_dset = ListData4Seg(data_folder=r'./Data', list_file='./datalist/extest_subject.txt', use_cw=use_cw)
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

# # load model
if use_cw:
    weight_path = './ckpts/UNet_seg' + '_cw'
    save_path = './results/UNet_seg' + '_cw'

weights_fname = 'weights.pth'
weights_fpath = os.path.join(weight_path, weights_fname)

if not os.path.exists(save_path):
    os.makedirs(save_path)

model.load_state_dict(torch.load(weights_fpath)['state_dict'])

model.eval()
with torch.no_grad():
    dices = []
    for batch_idx, (inputs, targets, flnm) in enumerate(test_loader):
        inputs = inputs.to(device).unsqueeze(1).float()
        targets = targets.to(device).unsqueeze(1).float()
        outputs = model(inputs)
        test_dice = 1 - dice_coefficient(outputs, targets)
        dices.append(test_dice.cpu().numpy())
        pred = np.squeeze(outputs.cpu().numpy())
        sio.savemat(save_path + '/' + flnm[0], {'pred_mask': pred})

    print(np.mean(dices), np.std(dices))

