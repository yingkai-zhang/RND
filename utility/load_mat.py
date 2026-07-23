import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random
import scipy.stats as stats
from torch.utils.data import DataLoader

from skimage import io
import cv2
import torch.distributions as tdist
import logging
logger = logging.getLogger('base')

from hdf5storage import loadmat

####################i##############################################################################

def load_tif_img(filepath):
    img = io.imread(filepath)
    img = img.astype(np.float32)
    #if type == 'gt':
    img = img/4096.

    return img

def is_tif_file(filename):
    return any(filename.endswith(extension) for extension in [".tif"])   

def load_mat_img(filepath, type='gt'):
    img = loadmat(filepath)
    if type == 'gt':
        img = img['label_normalized_hsi']
        img = img.transpose(2, 0, 1)
    else:
        img = img['lowlight_normalized_hsi']
        img = (img - img.min())
        img = img.transpose(2, 0, 1)

    return img



class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


class NoiseModel:
    def __init__(self, model='PGRC'):
        super().__init__()

        # self.param_dir = '../calibration/camera_params'#os.path.join('camera_params', 'V4')
        self.param_dir = './camera_params'

        print('[i] NoiseModel with {}'.format(self.param_dir))
        print('[i] using noise model {}'.format(model))
        
        self.camera_params = {}
        self.camera_K = {}

        # for band in range(18, 118, 3):
        for band in range(18, 208, 3):
            if band <= 118:
                _band = band
            else:
                _band = band - 102
            self.camera_params[_band] = np.load(os.path.join(self.param_dir, 'band_{}_params.npy'.format(_band)), allow_pickle=True).item()
            self.camera_K[_band] = np.load(os.path.join(self.param_dir.replace('camera_params', 'K'), 'color_{}.npy'.format(_band)), allow_pickle=True).item()

        self.model = model
        
    def _sample_params(self, band):
        Q_step = 1

        saturation_level = 4096
        profiles = ['Profile-1']

        camera_params = self.camera_params[band]
        camera_K = self.camera_K[band]

        G_shape = np.random.choice(camera_params['G_shape'])
        profile = np.random.choice(profiles)
        camera_params = camera_params[profile]

        # log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
        # log_K = np.random.uniform(low=np.log(1e-1), high=np.log(30))
        log_K = np.log(camera_K['K'])
        
        log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * 1 +\
             camera_params['g_scale']['slope'] * log_K + camera_params['g_scale']['bias']
        
        log_G_scale = np.random.standard_normal() * camera_params['G_scale']['sigma'] * 1 +\
             camera_params['G_scale']['slope'] * log_K + camera_params['G_scale']['bias']

        log_R_scale = np.random.standard_normal() * camera_params['R_scale']['sigma'] * 1 +\
             camera_params['R_scale']['slope'] * log_K + camera_params['R_scale']['bias']

        log_C_scale = np.random.standard_normal() * camera_params['C_scale']['sigma'] * 1 +\
             camera_params['C_scale']['slope'] * log_K + camera_params['C_scale']['bias']

        K = np.exp(log_K)
        g_scale = np.exp(log_g_scale)
        G_scale = np.exp(log_G_scale)
        R_scale = np.exp(log_R_scale)
        C_scale = np.exp(log_C_scale)

        # ratio = 10 #np.random.uniform(low=100, high=300)
        # ratio = np.random.uniform(low=1, high=300)

        return np.array([K, g_scale, G_scale, G_shape, R_scale, C_scale, Q_step, saturation_level]) 

    def __call__(self, input, ratio=10, params=None):
        output = []
        # for idx, band in enumerate(range(18, 118, 3)):
        for idx, band in enumerate(range(18, 208, 3)):
            if band <= 118:
                _band = band
            else:
                _band = band - 102

            y = input[idx:idx+1]
            if params is None:
                K, g_scale, G_scale, G_shape, R_scale, C_scale, Q_step, saturation_level = self._sample_params(_band)
            else:
                K, g_scale, G_scale, G_shape, R_scale, C_scale, Q_step, saturation_level = params

            y = y * saturation_level
            y = y / ratio
            
            if 'P' in self.model:
                z = np.random.poisson(y / K).astype(np.float32) * K
            elif 'p' in self.model:
                z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(K * y, 1e-10))
            else:
                z = y

            if 'g' in self.model:
                z = z + np.random.randn(*y.shape).astype(np.float32) * np.maximum(g_scale, 1e-10) # Gaussian noise            
            elif 'G' in self.model:
                z = z + stats.tukeylambda.rvs(G_shape, loc=0, scale=G_scale, size=y.shape).astype(np.float32) # Tukey Lambda 

            # if 'B' in self.model:
            #     z = self.add_color_bias(z, color_bias=color_bias)

            if 'R' in self.model:
                z = self.add_banding_noise(z, scale=R_scale, RC='R')
            
            if 'C' in self.model:
                z = self.add_banding_noise(z, scale=C_scale, RC='C')

            if 'U' in self.model:
                z = z + np.random.uniform(low=-0.5*Q_step, high=0.5*Q_step)     

            # z = z * ratio
            z = z / saturation_level
            output.append(z)

        return np.concatenate(output)

    def add_color_bias(self, img, color_bias):
        img = img + color_bias.reshape((4,1,1))
        return img

    def add_banding_noise(self, img, scale, RC=None):
        if RC == 'R':
            img = img + np.random.randn(1, img.shape[1], 1).astype(np.float32) * scale
        elif RC == 'C':
            img = img + np.random.randn(1, 1, img.shape[2]).astype(np.float32) * scale
        return img



class DataLoaderTrainLHSI(Dataset):
    def __init__(self, data_dir, ratio=15, img_options=None, target_transform=None,use2d=True,repeat=5):
        super(DataLoaderTrainLHSI, self).__init__()
        logger.info('==> DataLoaderTrainLHSI')

        self.target_transform = target_transform

        self.noise_model = NoiseModel()

        # clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        # noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))
        
        # self.clean_filenames = [os.path.join(data_dir, 'gt', x)          for x in clean_files if is_tif_file(x)]
        # self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x)     for x in noisy_files if is_tif_file(x)]

        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.clean_filenames = [os.path.join(data_dir, 'data', 'gt', line.strip()) for line in f]
        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.noisy_filenames = [os.path.join(data_dir, 'data', 'input', line.strip()) for line in f]

        self.clean = [torch.from_numpy(np.float32(load_mat_img(self.clean_filenames[index], 'gt'))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_mat_img(self.noisy_filenames[index], 'input'))) for index in range(len(self.noisy_filenames))]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.ratio = ratio
        self.use2d=use2d
        self.repeat =repeat

    def __len__(self):
        return self.tar_size*self.repeat

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]

        noisy1 = self.noise_model(clean, ratio=int(self.ratio))
        noisy1 = torch.from_numpy(noisy1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        # clean = torch.clamp(clean, 0, 1)
        # noisy = torch.clamp(noisy, 0, 1)
        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio
        noisy1 = noisy1[:, r:r + ps, c:c + ps] * self.ratio

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        noisy1 = getattr(augment, apply_trans)(noisy1)        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]
            noisy1 = noisy1[None,...]

        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        noisy1 = torch.clamp(noisy1, 0, 1)

        return  noisy,clean, noisy1,self.ratio#, clean_filename, noisy_filename
    


class DataLoaderTrainLHSIV1(Dataset):
    def __init__(self, data_dir, ratio=15, img_options=None, target_transform=None,use2d=True,repeat=5):
        super(DataLoaderTrainLHSIV1, self).__init__()
        logger.info('==> DataLoaderTrainLHSI')

        self.target_transform = target_transform

        # clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        # noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))
        
        # self.clean_filenames = [os.path.join(data_dir, 'gt', x)          for x in clean_files if is_tif_file(x)]
        # self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x)     for x in noisy_files if is_tif_file(x)]

        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.clean_filenames = [os.path.join(data_dir, 'data', 'gt', line.strip()) for line in f]
        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.noisy_filenames = [os.path.join(data_dir, 'data', 'input', line.strip()) for line in f]

        self.clean = [torch.from_numpy(np.float32(load_mat_img(self.clean_filenames[index], 'gt'))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_mat_img(self.noisy_filenames[index], 'input'))) for index in range(len(self.noisy_filenames))]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.ratio = ratio
        self.use2d=use2d
        self.repeat =repeat

    def __len__(self):
        return self.tar_size*self.repeat

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        # clean = torch.clamp(clean, 0, 1)
        # noisy = torch.clamp(noisy, 0, 1)
        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]

        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)

        return  noisy,clean,self.ratio#, clean_filename, noisy_filename



##################################################################################################
class DataLoaderValLHSI(Dataset):
    def __init__(self, data_dir, ratio=15, target_transform=None,use2d=True):
        super(DataLoaderValLHSI, self).__init__()
        logger.info('==> DataLoaderValLHSI')

        self.target_transform = target_transform

        # clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        # noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))

        # self.clean_filenames = [os.path.join(data_dir, 'gt', x)      for x in clean_files if is_tif_file(x)]
        # self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x) for x in noisy_files if is_tif_file(x)]

        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            self.clean_filenames = [os.path.join(data_dir, 'data', 'gt', line.strip()) for line in f]
        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            self.noisy_filenames = [os.path.join(data_dir, 'data', 'input', line.strip()) for line in f]

        self.clean = [torch.from_numpy(np.float32(load_mat_img(self.clean_filenames[index], 'gt'))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_mat_img(self.noisy_filenames[index], 'input'))) for index in range(len(self.noisy_filenames))]
        

        self.tar_size = len(self.clean_filenames)  
        self.ratio = ratio
        self.use2d = use2d

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1].split('.')[0]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1].split('.')[0]
       
        ps = 256
        r = clean.shape[1]//2-ps//2
        c = clean.shape[2]//2-ps//2
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio
        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        # print('clean shape:', clean.shape)
        # print('noisy shape:', noisy.shape)
        return  noisy,clean, clean_filename, noisy_filename