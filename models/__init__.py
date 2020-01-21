import apex
from apex import amp
import torch
import torch.nn as nn
import numpy as np 
from tqdm import tqdm 
import pandas as pd 
from pytorch_toolbelt.inference import tta as pytta
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from .resnet import ResNet 
from .utils import *
# from .metric import accuracy


def get_model(cfg):
    model = ResNet(model_name=cfg.TRAIN.MODEL, num_classes=cfg.TRAIN.NUM_CLASSES)
    return model

def train_loop(_print, cfg, model, train_loader, criterion, valid_loader,\
                valid_criterion, optimizer, scheduler, start_epoch, best_metric):

    tb = SummaryWriter() #for visualization
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        model.train()
        _print(f"Epoch {epoch+1}")

        losses = AverageMeter()
        top1 = AverageMeter()
        tbar = tqdm(train_loader)

        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()

            #calculate loss
            output = model(image)
            loss = criterion(output, target)
            acc = accuracy(output, target)

            #gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS #normalize the loss

            if cfg.SYSTEM.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()   #backward pass

            if (i + 1) % cfg.OPT.GD_STEPS == 0: #wait for several backward steps
                scheduler(optimizer, i, epoch, None) # Cosine LR Scheduler
                optimizer.step()    
                optimizer.zero_grad() #reset gradient

            #record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            top1.update(acc[0], image.size(0))
            tbar.set_description("Train top1: %.3f, loss: %.3f, learning rate: %.6f" % (top1.avg, losses.avg, optimizer.param_groups[-1]['lr']))

            #tensorboard
            tb.add_scalar('Loss', losses.avg, epoch)
            tb.add_scalar('Accuracy', top1.avg, epoch)
            tb.add_scalar('Lr', optimizer.param_groups[-1]['lr'], epoch)

        _print("Train top1: %.3f, loss: %.3f, learning rate: %.6f" % (top1.avg, losses.avg, optimizer.param_groups[-1]['lr']))   
        top1 = valid_model(_print, cfg, model, valid_loader, valid_criterion)
        is_best = top1 > best_metric
        best_metric = max(top1, best_metric)
        tb.add_scalar('Valid', top1, epoch) #tensorboard

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": cfg.EXP,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")

def valid_model(_print, cfg, model, valid_loader, valid_criterion, tta = False):
    losses = AverageMeter()
    top1 = AverageMeter()

    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)

    model.eval()
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            output = model(image)

            loss = valid_criterion(output, target)
            acc = accuracy(output, target)

            losses.update(loss.item(), image.size(0))
            top1.update(acc[0], image.size(0))          

    _print("Valid top1: %.3f, loss: %.3f" % (top1.avg, losses.avg))
    return top1.avg.data.cpu().numpy()[0]


if __name__ == "__main__":
    # losses = AverageMeter()
    pass