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

####################i##############################################################################

def get_aug_param_torch_single(b=8):
    aug = torch.zeros(b)
    r = np.random.randint(2) * 0.25 + 0.25
    u = r
    if np.random.randint(4):
        aug = torch.clamp(torch.randn(b) * r, 0, 4*u)
    
    # print(aug.shape)
    aug = aug.unsqueeze(1).repeat(1, 34)
    # print(aug.shape)

    # 保证非负
    daug, _ = torch.min(aug, dim=0)
    daug[daug>0] = 0
    aug = (1+aug) / (1+daug) - 1
    return aug

def get_aug_param_torch_all(b=8):
    aug = torch.zeros(b)
    r = np.random.randint(2) * 0.25 + 0.25
    u = r
    if np.random.randint(4):
        aug = torch.clamp(torch.randn(b) * r, 0, 4*u)

    aug = aug.unsqueeze(1).repeat(1, 34)

    for i in range(b):
        band = random.randint(0,33)
        for j in range(0, 34):
            if j != band:
                aug[i,j] = torch.clamp((1+torch.randn(1)*r) * (1+aug[i,band]) - 1, 0, 4*u)
    
    # print(aug.shape)
    # aug = aug.unsqueeze(1).repeat(1, 34)
    # print(aug.shape)

    # 保证非负
    daug, _ = torch.min(aug, dim=0)
    daug[daug>0] = 0
    aug = (1+aug) / (1+daug) - 1
    return aug

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


class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        ndim = rgb_gt.dim()
        if ndim == 5:
            rgb_gt = rgb_gt.squeeze(1)
            rgb_noisy = rgb_noisy.squeeze(1)

        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        if ndim == 5:
            rgb_gt = rgb_gt.unsqueeze(1)
            rgb_noisy = rgb_noisy.unsqueeze(1)

        return rgb_gt, rgb_noisy

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

def load_tif_img(filepath):
    img = io.imread(filepath)
    img = img.astype(np.float32)
    #if type == 'gt':
    img = img/4096.

    return img

def is_tif_file(filename):
    return any(filename.endswith(extension) for extension in [".tif"])   

class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, ratio=50, img_options=None, target_transform=None,use2d=True,repeat=5):
        super(DataLoaderTrain, self).__init__()
        logger.info('==> DataLoaderTrain')

        self.target_transform = target_transform

        # clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        # noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))
        
        # self.clean_filenames = [os.path.join(data_dir, 'gt', x)          for x in clean_files if is_tif_file(x)]
        # self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x)     for x in noisy_files if is_tif_file(x)]

        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.clean_filenames = [os.path.join(data_dir, 'gt', line.strip()) for line in f]
        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), line.strip()) for line in f]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        
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

        return  noisy,clean, self.ratio#, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, data_dir, ratio=50, target_transform=None,use2d=True):
        super(DataLoaderVal, self).__init__()
        logger.info('==> DataLoaderVal')

        self.target_transform = target_transform

        # clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        # noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))

        # self.clean_filenames = [os.path.join(data_dir, 'gt', x)      for x in clean_files if is_tif_file(x)]
        # self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x) for x in noisy_files if is_tif_file(x)]

        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            self.clean_filenames = [os.path.join(data_dir, 'gt', line.strip()) for line in f]
        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), line.strip()) for line in f]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        

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
       
        ps = 512
        r = clean.shape[1]//2-ps//2
        c = clean.shape[2]//2-ps//2
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio
        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        return  noisy,clean, clean_filename, noisy_filename


if __name__ == '__main__':
    rgb_dir = '/media/lmy/LMY/aaai/real_dataset'
    ratio = 50
    train_dir = '/media/lmy/LMY/aaai/train_real/'
    img_options ={}
    img_options['patch_size'] = 128
    #train_dataset = DataLoaderTrain(train_dir,50,img_options=img_options)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=1, shuffle=True,
    #                           num_workers=1)
    test_dir= '/media/lmy/LMY/aaai/test_real/'
    dataset = DataLoaderVal(test_dir, ratio, None)
    
    # print(len(dataset))
   
    train_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    #print(iter(train_loader).next())
    for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(batch_idx,inputs.shape)
            band =20
            inputs = inputs.numpy()
            targets = targets.numpy()
            cv2.imwrite('tnoisy_'+'_band'+str(band)+'.png',inputs[0,band]*255)
            cv2.imwrite('tgt_'+'_band'+str(band)+'.png',targets[0,band]*255)
            break
    
class NoiseModel:
    def __init__(self, model='PGRC'):
        super().__init__()

        self.param_dir = './camera_params'

        logger.info('==> NoiseModel with {}'.format(self.param_dir))
        logger.info('==> using noise model {}'.format(model))
        
        self.camera_params = {}
        self.camera_K = {}

        for band in range(18, 118, 3):
            self.camera_params[band] = np.load(os.path.join(self.param_dir, 'band_{}_params.npy'.format(band)), allow_pickle=True).item()
            self.camera_K[band] = np.load(os.path.join(self.param_dir.replace('camera_params', 'K'), 'color_{}.npy'.format(band)), allow_pickle=True).item()

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
        for idx, band in enumerate(range(18, 118, 3)):
            y = input[idx:idx+1]
            if params is None:
                K, g_scale, G_scale, G_shape, R_scale, C_scale, Q_step, saturation_level = self._sample_params(band)
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
        

##################################################################################################
class DataLoaderTrainNoiseKnownMultiRatio(Dataset):
    def __init__(self, data_dir, ratio=[20,50,100], img_options=None, target_transform=None,use2d=True,repeat=5):
        super(DataLoaderTrainNoiseKnownMultiRatio, self).__init__()
        logger.info('==> DataLoaderTrainNoiseKnownMultiRatio')

        self.noise_model = NoiseModel()

        self.target_transform = target_transform

        with open(os.path.join(data_dir, 'train_scene.txt'), 'r') as f:
            self.filenames = [line.strip().split('.')[0] for line in f]

        self.data = {}

        for filename in self.filenames:
            print(filename)
            self.data[filename] = {}
            self.data[filename]['gt'] = np.float32(load_tif_img(os.path.join(data_dir, 'gt', filename + '.tif')))

        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.trainlist = [line.strip() for line in f]
        
        self.img_options=img_options

        self.tar_size = len(self.trainlist)  # get the size of target
        self.use2d=use2d
        self.repeat =repeat

    def __len__(self):
        return self.tar_size*self.repeat

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        name, ratio = self.trainlist[tar_index].split('_')
        clean = self.data[name]['gt']

        noisy = self.noise_model(clean, ratio=int(ratio))
        
        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy) * int(ratio)

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]

        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)

        return  noisy, clean, ratio, name
    
class DataLoaderValNoiseKnownMultiRatio(Dataset):
    def __init__(self, data_dir, ratio=[20,50,100], target_transform=None,use2d=True):
        super(DataLoaderValNoiseKnownMultiRatio, self).__init__()
        logger.info('==> DataLoaderValNoiseKnownMultiRatio')

        self.target_transform = target_transform

        with open(os.path.join(data_dir, 'test_scene.txt'), 'r') as f:
            self.filenames = [line.strip().split('.')[0] for line in f]

        self.data = {}

        for filename in self.filenames:
            self.data[filename] = {}
            self.data[filename]['gt'] = torch.from_numpy(np.float32(load_tif_img(os.path.join(data_dir, 'gt', filename + '.tif'))))
            for r in ratio:
                self.data[filename][str(r)] = torch.from_numpy(np.float32(load_tif_img(os.path.join(data_dir, 'noise_known_input{}'.format(str(r)), filename+'.tif'))))

        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            self.testlist = [line.strip() for line in f]
        

        self.tar_size = len(self.testlist)
        self.use2d = use2d

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        name, ratio = self.testlist[tar_index].split('_')
        clean = self.data[name]['gt']
        noisy = self.data[name][ratio]
       
        ps = 512
        r = clean.shape[1]//2-ps//2
        c = clean.shape[2]//2-ps//2
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * int(ratio)
        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        return  noisy, clean, ratio, name

##################################################################################################
class DataLoaderTrainNoiseKnown(Dataset):
    def __init__(self, data_dir, ratio=50, img_options=None, target_transform=None,use2d=True,repeat=5):
        super(DataLoaderTrainNoiseKnown, self).__init__()
        logger.info('==> DataLoaderTrainNoiseKnown')

        self.noise_model = NoiseModel()

        self.target_transform = target_transform

        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.clean_filenames = [os.path.join(data_dir, 'gt', line.strip()) for line in f]

        self.clean = [np.float32(load_tif_img(self.clean_filenames[index])) for index in range(len(self.clean_filenames))]
        
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
        # clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]

        noisy = self.noise_model(clean, ratio=self.ratio)
        
        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy) * self.ratio

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]

        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)

        return  noisy,clean, self.ratio#, clean_filename, noisy_filename
    
class DataLoaderValNoiseKnown(Dataset):
    def __init__(self, data_dir, ratio=50, target_transform=None,use2d=True):
        super(DataLoaderValNoiseKnown, self).__init__()
        logger.info('==> DataLoaderValNoiseKnown')

        self.target_transform = target_transform

        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            self.clean_filenames = [os.path.join(data_dir, 'gt', line.strip()) for line in f]
        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            self.noisy_filenames = [os.path.join(data_dir, 'noise_known_input{}'.format(ratio), line.strip()) for line in f]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        

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
       
        ps = 512
        r = clean.shape[1]//2-ps//2
        c = clean.shape[2]//2-ps//2
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio
        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        return  noisy,clean, clean_filename, noisy_filename
    
class DataLoaderTrainNoiseUnKnownOnline(Dataset):
    def __init__(self, data_dir, ratio=50, img_options=None, target_transform=None,use2d=True,repeat=5):
        super(DataLoaderTrainNoiseUnKnownOnline, self).__init__()
        logger.info('==> DataLoaderTrainNoiseUnKnownOnline')

        self.noise_model = NoiseModel()

        self.target_transform = target_transform

        # clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        # noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))
        
        # self.clean_filenames = [os.path.join(data_dir, 'gt', x)          for x in clean_files if is_tif_file(x)]
        # self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x)     for x in noisy_files if is_tif_file(x)]

        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.clean_filenames = [os.path.join(data_dir, 'gt', line.strip()) for line in f]
        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), line.strip()) for line in f]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        
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

        noisy1 = self.noise_model(clean, ratio=self.ratio)
        noisy1 = torch.from_numpy(noisy1)

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

        return  noisy,clean, noisy1, self.ratio#, clean_filename, noisy_filename


class DataLoaderTrainNoiseUnKnownOnlineMultiRatioRelease(Dataset):
    def __init__(self, data_dir, ratio=[20,50,100], img_options=None, target_transform=None,use2d=True,repeat=5):
        super(DataLoaderTrainNoiseUnKnownOnlineMultiRatioRelease, self).__init__()
        logger.info('==> DataLoaderTrainNoiseUnKnownOnlineMultiRatioRelease')

        self.noise_model = NoiseModel()

        self.target_transform = target_transform

        with open(os.path.join(data_dir, 'train_scene.txt'), 'r') as f:
            self.filenames = [line.strip().split('.')[0] for line in f]

        self.data = {}

        for filename in self.filenames:
            print(filename)
            self.data[filename] = {}
            # self.data[filename]['gt'] = torch.from_numpy(np.float32(load_tif_img(os.path.join(data_dir, 'gt', filename + '.tif'))))
            # for r in ratio:
            #     self.data[filename][str(r)] = torch.from_numpy(np.float32(load_tif_img(os.path.join(data_dir, 'input{}'.format(str(r)), filename+'.tif'))))
            self.data[filename]['gt'] = np.float32(load_tif_img(os.path.join(data_dir, 'gt', filename + '.tif')))
            for r in ratio:
                self.data[filename][str(r)] = np.float32(load_tif_img(os.path.join(data_dir, 'input{}'.format(str(r)), filename+'.tif')))


        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.trainlist = [line.strip() for line in f]
        
        self.img_options=img_options

        self.tar_size = len(self.trainlist)  # get the size of target
        self.use2d=use2d
        self.repeat =repeat

    def __len__(self):
        return self.tar_size*self.repeat

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        name, ratio = self.trainlist[tar_index].split('_')
        clean = self.data[name]['gt']
        noisy = self.data[name][ratio]

        noisy1 = self.noise_model(clean, ratio=int(ratio))
        noisy1 = torch.from_numpy(noisy1)
        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        # clean = torch.clamp(clean, 0, 1)
        # noisy = torch.clamp(noisy, 0, 1)
        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]

        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * int(ratio)
        noisy1 = noisy1[:, r:r + ps, c:c + ps] * int(ratio)

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

        return  noisy,clean, noisy1, ratio, name
    

##################################################################################################
class DataLoaderValMultiRatioRelease(Dataset):
    def __init__(self, data_dir, ratio=[20,50,100], target_transform=None,use2d=False):
        super(DataLoaderValMultiRatioRelease, self).__init__()
        logger.info('==> DataLoaderValMultiRatioRelease')

        self.target_transform = target_transform

        with open(os.path.join(data_dir, 'test_scene.txt'), 'r') as f:
            self.filenames = [line.strip().split('.')[0] for line in f]

        self.data = {}

        for filename in self.filenames:
            self.data[filename] = {}
            self.data[filename]['gt'] = torch.from_numpy(np.float32(load_tif_img(os.path.join(data_dir, 'gt', filename + '.tif'))))
            for r in ratio:
                self.data[filename][str(r)] = torch.from_numpy(np.float32(load_tif_img(os.path.join(data_dir, 'input{}'.format(str(r)), filename+'.tif'))))

        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            self.testlist = [line.strip() for line in f]
        

        self.tar_size = len(self.testlist)
        self.use2d = use2d

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        name, ratio = self.testlist[tar_index].split('_')
        clean = self.data[name]['gt']
        noisy = self.data[name][ratio]
       
        ps = 512
        r = clean.shape[1]//2-ps//2
        c = clean.shape[2]//2-ps//2

        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * int(ratio)
        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        return  noisy, clean, ratio, name
    


class DataLoaderTrainMultiRatioRelease(Dataset):
    def __init__(self, data_dir, ratio=[20,50,100], img_options=None, target_transform=None,use2d=True,repeat=5):
        super(DataLoaderTrainMultiRatioRelease, self).__init__()
        logger.info('==> DataLoaderTrainMultiRatioRelease')

        self.target_transform = target_transform

        with open(os.path.join(data_dir, 'train_scene.txt'), 'r') as f:
            self.filenames = [line.strip().split('.')[0] for line in f]

        self.data = {}

        for filename in self.filenames:
            print(filename)
            self.data[filename] = {}
            self.data[filename]['gt'] = torch.from_numpy(np.float32(load_tif_img(os.path.join(data_dir, 'gt', filename + '.tif'))))
            for r in ratio:
                self.data[filename][str(r)] = torch.from_numpy(np.float32(load_tif_img(os.path.join(data_dir, 'input{}'.format(str(r)), filename+'.tif'))))

        with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
            self.trainlist = [line.strip() for line in f]
        
        self.img_options=img_options

        self.tar_size = len(self.trainlist)  # get the size of target
        self.use2d=use2d
        self.repeat =repeat

    def __len__(self):
        return self.tar_size*self.repeat

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        name, ratio = self.trainlist[tar_index].split('_')
        clean = self.data[name]['gt']
        noisy = self.data[name][ratio]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * int(ratio)

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]

        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)

        return  noisy, clean, ratio, name