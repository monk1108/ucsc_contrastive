import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import date
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from logger import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

def fine_tune_ViT(model, train_dataloader, val_dataloader, args, dev, multi_label_data=False):
    tb_writer = SummaryWriter('./myExperiment/' + args.name)
    lr = args.lr
    n_epochs = args.e
    wd = args.wd
    name = args.name
    eps = args.eps

    today = date.today()
    today = today.isoformat()
    ckpt_dir = './checkpoints/' + today + '-' + name + '/'
    # ckpt_dir = './checkpoints/2022-04-20-1/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
   
    
    logger = get_logger(ckpt_dir + 'exp.log')
    logger.info('parameter setting:')
    logger.info(vars(args))
    logger.info('start training!')

    model = model.to(dev) 
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99), eps=eps, weight_decay=wd)
    # optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9,0.99), eps=1e-6, weight_decay=0.2)
    
    epoch = 0
    # checkpoint = torch.load(ckpt_dir + 'model_newest.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    
    ### Different loss criterion for multi-label and single-label data
    if multi_label_data:
        # weight = torch.ones([14])
        weight = torch.tensor([8.7, 39.4, 7.4, 4.6, 18.4, 16.7, 77.35, 20.1, 23.0, 47.7, 43.6, 65.5,\
            32.1, 492.9])
        # criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        criterion = nn.BCEWithLogitsLoss()
    else: # single-label data (standard)
        criterion = nn.CrossEntropyLoss()

    for e in range(epoch, epoch + n_epochs):
        print('--------------------------------------------------------------------------')
        print('epoch: ', str(e + 1))
        model.train()
        train_loss = 0.
        train_acc = 0.
        val_acc = 0.
        train_auc = 0.
        val_auc = 0.
        count = 0

        for img, tgt in tqdm(train_dataloader):
            # print(batch_i)
            # yaoyinuo 2022.4.25
            img, tgt = img.to(dev), tgt.to(dev)
            img = img.squeeze(1)


            output = model(img) 
            # logits = classifier(features.float()) # [batch_size, n_classes] 
            
            optimizer.zero_grad()
            # optimizer_classifier.zero_grad()
            loss = criterion(output.type(torch.FloatTensor), tgt.type(torch.FloatTensor))

            loss.backward()
            optimizer.step()
            # optimizer_classifier.step()
            train_loss += loss.item()

            tb_writer.add_scalar('training loss', loss, e * len(train_dataloader) + count)
            
            count += 1

        #  3. Evaluate on valdiation data  
        model.eval()
        val_loss = 0.


        outGT = torch.FloatTensor().to(dev)
        outPRED = torch.FloatTensor().to(dev)

        for img, tgt in tqdm(val_dataloader):
            # img, tgt = img.to(dev), tgt.to(dev)

            # img = torch.stack([img], dim=0)
            img, tgt = img.to(dev), tgt.to(dev)
            img = img.squeeze(1)

            outGT = torch.cat((outGT, tgt), 0)    # concatenate all the batches
            with torch.no_grad():
                output = model(img)
                outPRED = torch.cat((outPRED, output), 0)
                loss = criterion(output.type(torch.FloatTensor), tgt.type(torch.FloatTensor))
                val_loss += loss.item()

                # pred = torch.argmax(logits, axis=1)
                # label = torch.argmax(tgt, axis=1)
        
        
        sgmd = nn.Sigmoid()
        outPRED = sgmd(outPRED)
        pred = outPRED > 0.5

        output_np = outPRED.detach().cpu().numpy()
        tgt_np = outGT.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        aucs = []
        accs = []

        for i in range(14):
            aucs.append(roc_auc_score(tgt_np[:, i], output_np[:, i]))
            accs.append(np.mean((pred[:, i] == tgt_np[:, i])))
        
        auc_mean = np.array(aucs).mean()
        acc_mean = np.array(accs).mean()
        print('-----------------------------------------------------------------------')
        print('Val Result AUC:')
        print(aucs)
        print('AUC Mean: {:.4f}'.format(auc_mean))
        print('Val Result ACCURACY:')
        print(accs)
        # print(aucs)
        print('ACC Mean: {:.4f}'.format(acc_mean))

        #  4. Log results and save model checkpoints 
        print("Epoch: {}/{}  TrainLoss: {:.5f}  ValLoss: {:.5f}".format(e+1, n_epochs, \
            train_loss/len(train_dataloader), val_loss/len(val_dataloader)))

        torch.save({
            'epoch': e+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss/len(train_dataloader),
            'val_loss': val_loss/len(val_dataloader),
            }, 
            (ckpt_dir + 'model_newest.pt'))

        logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_loss={:.5f}\n \
            train_acc={:.3f}\t val_acc={:.3f}\t train_auc={:.3f}\t val_auc={:.3f}'.format(e+1, n_epochs, \
                train_loss/len(train_dataloader), val_loss/len(val_dataloader),
                train_acc/len(train_dataloader), val_acc/len(val_dataloader), \
                    train_auc/len(train_dataloader), val_auc/len(val_dataloader)))
    
    logger.info('finish training!')
    return model



def test(model, test_dataloader, args, dev, multi_label_data=False):
    lr = args.lr
    # n_epochs = args.e
    wd = args.wd
    name = args.name
    eps = args.eps

    today = date.today()
    today = today.isoformat()
    # ckpt_dir = './checkpoints/' + today + '-' + name + '/'
    ckpt_dir = './checkpoints/2022-04-24-stack/'

    model = model.to(dev) 
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99), eps=eps, weight_decay=wd)
    # criterion = nn.BCELoss()

    checkpoint = torch.load(ckpt_dir + 'model_newest.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    # val_loss = 0.
    outGT = torch.FloatTensor().to(dev)
    outPRED = torch.FloatTensor().to(dev)
    sgmd = nn.Sigmoid()

    for img, tgt in tqdm(test_dataloader):
        img, tgt = img.to(dev), tgt.to(dev)
        img = img.squeeze(1)
        outGT = torch.cat((outGT, tgt), 0)    # concatenate all the batches
        with torch.no_grad():
            output = model(img)
            output = sgmd(output)
            outPRED = torch.cat((outPRED, output), 0)

            output_np = output.detach().cpu().numpy()
            tgt_np = tgt.detach().cpu().numpy()
            auc = roc_auc_score(tgt_np, output_np, average='micro')

    sgmd = nn.Sigmoid()
    outPRED = sgmd(outPRED)
    pred = outPRED > 0.5
    # acc = (pred == outGT).detach().cpu().numpy()
    # acc3 = np.mean(acc) * 100.
    # print(acc3)

    output_np = outPRED.detach().cpu().numpy()
    tgt_np = outGT.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    aucs = []
    accs = []

    for i in range(14):
        aucs.append(roc_auc_score(tgt_np[:, i], output_np[:, i]))
        accs.append(np.mean((pred[:, i] == tgt_np[:, i])))
    
    auc_mean = np.array(aucs).mean()
    acc_mean = np.array(accs).mean()
    print('-----------------------------------------------------------------------')
    print('Test Result AUC:')
    print(aucs)
    print('AUC Mean: {:.4f}'.format(auc_mean))
    print('Test Result ACCURACY:')
    print(accs)
    # print(aucs)
    print('ACC Mean: {:.4f}'.format(acc_mean))

