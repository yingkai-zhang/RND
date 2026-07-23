import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
from hsi_setup_ratio import Engine, train_options, make_dataset
import logging

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    # print(opt)

    basedir = os.path.join('result', opt.arch)

    if not os.path.exists(basedir):
        os.makedirs(basedir)

    if opt.resume:
        mkdir_and_rename(os.path.join(basedir, opt.prefix))

    setup_logger('base', os.path.join(basedir, opt.prefix), 'train_' + opt.prefix, level=logging.INFO,
                        screen=True, tofile=True)
    logger = logging.getLogger('base')
    # logger.info(dict2str(opt))
    logger.info(opt)
    

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.net.use_2dconv)


    """Test-Dev"""
    basefolder = opt.rootdir
    # testlist = opt.testlist
  
    mat_datasets = DataLoaderValMultiRatioRelease(basefolder, [20,50,100], None,use2d=engine.get_net().use_2dconv)
    # print(len(mat_datasets))
    logger.info('The number of test samples is: %d' % len(mat_datasets))
    # print('loading finished')
    logger.info('loading finished')

 
    mat_loader = DataLoader(
        mat_datasets,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda     )  
      
    strart_time = time.time()
    # engine.test(mat_loader, basefolder)
    engine.test(mat_loader)
    end_time = time.time()
    test_time = end_time-strart_time
    # print('cost-time: ',(test_time/15))
    logger.info('cost-time: %f'%(test_time/15))
        
