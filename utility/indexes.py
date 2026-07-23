import numpy as np
import torch
# from skimage.measure import compare_ssim, compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from functools import partial
import torch.nn.functional as F


class Bandwise(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex
    
class Bandwise_torch(object): 
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data)
            y = torch.squeeze(Y[...,ch,:,:].data)
            index = self.index_fn(x, y)
            bwindex.append(index.cpu().numpy())
        return bwindex

def KLDivergenceLoss(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    loss = F.kl_div(p.log(), q, reduction='batchmean')
    return loss


cal_bwssim = Bandwise(compare_ssim)
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))
cal_bwkl = Bandwise_torch(KLDivergenceLoss)


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)    
    return np.mean(np.real(np.arccos(tmp)))


def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    return psnr, ssim, sam
