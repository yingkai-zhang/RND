import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
import datetime
import time
from hsi_setup_ratio import Engine, train_options, make_dataset
#os.environ["WANDB_MODE"] ='offline'
import logging

if __name__ == '__main__':
    """Training settings"""
    
    parser = argparse.ArgumentParser(
    description='Hyperspectral Image Denoising (Real noise)')
    opt = train_options(parser)
    # print(opt)

    basedir = os.path.join('result', opt.arch)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    if not opt.resume:
        mkdir_and_rename(os.path.join(basedir, opt.prefix))
    
    if not os.path.exists(os.path.join(basedir, opt.prefix, 'ckpt')):
        os.makedirs(os.path.join(basedir, opt.prefix, 'ckpt'))

    if not os.path.exists(os.path.join(basedir, opt.prefix, 'val_images')):
        os.makedirs(os.path.join(basedir, opt.prefix, 'val_images'))

    setup_logger('base', os.path.join(basedir, opt.prefix), 'train_' + opt.prefix, level=logging.INFO,
                        screen=True, tofile=True)
    logger = logging.getLogger('base')
    # logger.info(dict2str(opt))
    logger.info(opt)
    
    img_options={}
    img_options['patch_size'] = 128

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    # train_dir = '/train_real/'
    basefolder = opt.rootdir
    # trainlist = opt.trainlist
    

    train_dataset = DataLoaderTrainNoiseKnownMultiRatio(basefolder,[20,50,100],img_options=img_options,use2d=engine.get_net().use_2dconv,repeat=opt.repeat)
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn, drop_last=True)
   
    # print('==> Preparing data..')
    logger.info('==> Preparing data..')


    """Test-Dev"""
    
    # testlist = opt.testlist
    
    mat_datasets = DataLoaderValNoiseKnownMultiRatio(basefolder, [20,50,100], None,use2d=engine.get_net().use_2dconv)
    
    mat_loader = DataLoader(
        mat_datasets,
        batch_size=1, shuffle=False,
        num_workers=8, pin_memory=opt.no_cuda, drop_last=False
    )      

    base_lr = opt.lr
    epoch_per_save = 20
    # adjust_learning_rate(engine.optimizer, opt.lr)
    # print('loading finished')
    logger.info('loading finished')

    best_psnr = 0
    best_epoch = 0
    # best_ssim = 0
    # best_sam = 0

    # from epoch 50 to 100
    # engine.epoch  = 0
    while engine.epoch < opt.total_epochs:
        # np.random.seed()

        # if engine.epoch == 200:
        
        #     adjust_learning_rate(engine.optimizer, base_lr*0.5)
          
        # if engine.epoch == 400:
            
        #     adjust_learning_rate(engine.optimizer, base_lr*0.1)

        
        engine.train(train_loader,mat_loader)
        
        if engine.epoch % opt.val_epoch == 0:

            # avg_psnr, avg_loss,avg_sam = engine.validate(mat_loader, 'real')
            avg_psnr, avg_loss,avg_sam = engine.test(mat_loader)
            

            # display_learning_rate(engine.optimizer)
            # # print('Latest Result Saving...')
            # logger.info('Latest Result Saving...')
            # model_latest_path = os.path.join(engine.basedir, engine.prefix, 'kpt', 'model_latest.pth')
            # engine.save_checkpoint(
            #     model_out_path=model_latest_path
            # )

            logger.info(
                    '# Validation # Average PSNR: {:.4e} Previous best Average PSNR: {:.4e} Previous best Average epoch: {}'.
                    format(avg_psnr, best_psnr, best_epoch))
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch = engine.epoch
                logger.info('Saving best average models!!!!!!!The best psnr is:{:4e}'.format(best_psnr))
                model_latest_path = os.path.join(engine.basedir, engine.prefix, 'ckpt', 'model_best.pth')
                engine.save_checkpoint(
                    model_out_path=model_latest_path
                )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

        engine.scheduler.step()
    # wandb.finish()
