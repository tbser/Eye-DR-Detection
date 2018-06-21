import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets as dsets
from torchvision import transforms as trans
import numpy as np
import torchvision.models as models
from my_resnet import my_model, my_model_256
import settings

logger = settings.getlogger("eval_256")


def get_paras512(model_path=None):
    val_trans = trans.Compose([
        trans.ToTensor()])
    test_data = dsets.ImageFolder(root='./dataset_pre/val/', transform=val_trans)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=3)

    model = models.resnet34(pretrained=True)
    in_features = model.fc.in_features
    num_class = len(test_loader.dataset.classes)
    model.avgpool = nn.AvgPool2d(10, 10)
    model.fc = nn.Linear(in_features, num_class)

    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model, test_loader


def get_paras128(model_path=None):
    val_trans = trans.Compose([
        trans.Resize(128),
        trans.ToTensor()])
    test_data = dsets.ImageFolder(root='./dataset_pre/val/', transform=val_trans)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=3)

    model = my_model(num_classes=len(test_loader.dataset.classes), pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model, test_loader


def get_paras256(model_path=None):
    val_trans = trans.Compose([
        trans.Resize(256),
        trans.ToTensor()])
    test_data = dsets.ImageFolder(root='./dataset_pre/val/', transform=val_trans)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=3)

    model = my_model_256(num_classes=len(test_loader.dataset.classes), pretrained=True)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model, test_loader


def eval_model(model, test_loader):
    all = np.zeros(shape=[5])
    hit = np.zeros(shape=[5])
    model.eval()
    with torch.no_grad():
        num_hit = 0
        total = len(test_loader.dataset)
        for batch_idx, (image, labels) in enumerate(test_loader):
            output = model(image)
            _, pred_label = output.data.max(dim=1)
            pred_label = np.copy(pred_label)
            truth_label = np.copy(labels.data)
            yi = truth_label[0]
            yi_ = pred_label[0]
            if yi == 0:
                all[0] += 1
                if yi_ == yi:
                    hit[0] += 1
            elif yi == 1:
                all[1] += 1
                if yi_ == yi:
                    hit[1] += 1
            elif yi == 2:
                all[2] += 1
                if yi_ == yi:
                    hit[2] += 1
            elif yi == 3:
                all[3] += 1
                if yi_ == yi:
                    hit[3] += 1
            elif yi == 4:
                all[4] += 1
                if yi_ == yi:
                    hit[4] += 1
            num_hit += (pred_label == truth_label).sum()
            logger.info('index:{}/{}'.format(batch_idx + 1, len(test_loader)))
        logger.info('numhit:{}'.format(num_hit))
        test_accuracy = (num_hit / total)
        logger.info("Testing Accuracy (num_hit/total): {:.2f}%".format(100. * test_accuracy))
        logger.info('class 0 acc:{:.2f}%, class 1 acc:{:.2f}%, class 2 acc: {:.2f}%, '
                    'class 3 acc: {:.2f}%, class 4 acc: {:.2f}%'.format(
                    hit[0] / all[0] * 100., hit[1] / all[1] * 100., hit[2] / all[2] * 100.,
                    hit[3] / all[3] * 100., hit[4] / all[4] * 100.))


if __name__ == '__main__':
    # model, test_loader = get_paras512()
    # model, test_loader = get_paras128(model_path='./model_best.pth.tar')
    model, test_loader = get_paras256(model_path='./model_best.pth.tar')
    eval_model(model, test_loader)
