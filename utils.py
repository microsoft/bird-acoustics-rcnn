# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import librosa
from joblib import Parallel, delayed
import multiprocessing
import h5py
import pickle as pk
import cv2
import os
import matplotlib.pyplot as plt
from fastprogress.fastprogress import progress_bar


class log_results(object):
    def __init__(self, file_name = 'log', results_dir = 'Results'):
        self.results_dir = results_dir
        self.fname = file_name
        
        if not os.path.exists(self.results_dir):
            os.makedirs(results_dir)
        
    def update(self, log):
        file_path = os.path.join(self.results_dir, self.fname)
        if isinstance(log, dict):
            pk.dump(log, open(file_path, 'ab'))
        else:
            print('log has to be in dictionary format')
            

class SaveBestModel(object):

    def __init__(self, monitor = np.inf, PATH = './currTorchModel.pt', 
                    verbose=False):

        self.monitor = monitor
        self.PATH = PATH
        self.verbose = verbose

    def check(self, model, currVal, comp='min'):
        if comp is 'min':
            if currVal < self.monitor:
                self.monitor = currVal
                torch.save(model.state_dict(), self.PATH)
                if self.verbose:
                    print('saving best model...')
        elif comp is 'max':
            if currVal > self.monitor:
                self.monitor = currVal
                torch.save(model.state_dict(), self.PATH)
                if self.verbose:
                    print('saving best model...')
                    

def normalize_mel_sp_slides(X, eps=1e-6):
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = _max
    norm_min = _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = (V - norm_min) / (norm_max - norm_min)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    return V


def mel_sp_slides_to_image(X, eps=1e-6, resize=False, nrow=224, ncol=224):
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    # cmap = plt.cm.jet
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=Xstd.min(), vmax=Xstd.max())

    # map the normalized data to colors
    # image is now RGBA (nrowxncolx4) 
    # last channel is alpha value for transparency, set to 1    
    image = cmap(norm(Xstd))
    if resize:
        return cv2.resize(
                    image[:,:,:3], (nrow, ncol), 
                    interpolation=cv2.INTER_LINEAR
                )
    else:
        return image[:,:,:,:3]
    

def mel_sp_to_image(X, eps=1e-6, nrow=224, ncol=224):
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    # cmap = plt.cm.jet
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=Xstd.min(), vmax=Xstd.max())

    # map the normalized data to colors
    # image is now RGBA (nrowxncolx4) 
    # last channel is alpha value for transparency, set to 1    
    image = cmap(norm(Xstd))
    return cv2.resize(image[:,:,:3], (nrow, ncol), 
                      interpolation=cv2.INTER_LINEAR
             )
    
    

def train_seq(model, train_loader, optimizer, epoch, device, verbose = 0,
            lr_schedule = None, weight = None, loss_fn = 'crossEnt'):
    """Training"""
    if lr_schedule is not None:
        optimizer = lr_schedule(optimizer, epoch)

    model.train()
    for batch_idx, (data, target) in enumerate(progress_bar(train_loader)):
        
        h_s = model.init_hidden(len(data))
        if isinstance(h_s, tuple):
            h_s = tuple([x.to(device) for x in h_s])
        else:
            h_s = h_s.to(device)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data, h_s)
        if loss_fn == 'crossEnt':
            criteria = nn.CrossEntropyLoss().cuda()
        elif loss_fn == 'bceLogit':
            criteria = nn.BCEWithLogitsLoss().cuda()
            
        loss = criteria(output, target)

        loss.backward()
        optimizer.step()

    if verbose>0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()


def evalModel_seq(data_loader, model, device, verbose=0, stochastic_pass = True,
                 compute_metrics=True, activationName = None,
                 loss_fn = 'crossEnt'):

    if stochastic_pass:
        model.train()
    else:
        model.eval()

    test_loss = 0
    predictions = []
    activations = []
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)
            
            h_s = model.init_hidden(len(data))
            if isinstance(h_s, tuple):
                h_s = tuple([x.to(device) for x in h_s])
            else:
                h_s =h_s.to(device)
                
            output = model(data, h_s)

            if compute_metrics:
                predictionClasses = output.argmax(dim=1, keepdim=True)
                if loss_fn == 'crossEnt':
                    criteria = nn.CrossEntropyLoss().cuda()
                    correct += predictionClasses.eq(target.view_as(predictionClasses)).sum().item()
                elif loss_fn == 'bceLogit':
                    criteria = nn.BCEWithLogitsLoss().cuda()
                    correct += predictionClasses.eq(target.argmax(dim=1).view_as(predictionClasses)).sum().item()

                test_loss += criteria(output, target).sum().item()
            else:
                softmaxed = F.softmax(output.cpu(), dim=1)
                predictions.extend(softmaxed.data.numpy())

    if compute_metrics:
        return test_loss, correct
    else:
        return predictions, activations
    

def test_seq(model, test_loader, device, verbose=0, activationName = None,
            loss_fn = 'crossEnt'):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0

    total_test_loss, total_corrections = evalModel_seq(test_loader, model, device=device,
                            verbose = verbose, 
                            stochastic_pass = False, compute_metrics = True, 
                            activationName = activationName, loss_fn = loss_fn)

    test_loss = total_test_loss/ len(test_loader) # loss function already averages over batch size
    test_acc = total_corrections / len(test_loader.dataset)
    if verbose>0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        print('{{"metric": "Eval - cross entropy Loss", "value": {}, "epoch": {}}}'.format(
            test_loss, epoch))
        print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
            100. * correct / len(test_loader.dataset), epoch))

    return test_loss, test_acc


def train(model, train_loader, optimizer, epoch, device, verbose = 0,
            lr_schedule = None, weight = None, loss_fn = 'crossEnt'):
    """Training"""
    if lr_schedule is not None:
        optimizer = lr_schedule(optimizer, epoch)

    model.train()
    for batch_idx, (data, target) in enumerate(progress_bar(train_loader)):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        if loss_fn == 'crossEnt':
            criteria = nn.CrossEntropyLoss().cuda()
        elif loss_fn == 'bceLogit':
            criteria = nn.BCEWithLogitsLoss().cuda()
            
        loss = criteria(output, target)

        loss.backward()
        optimizer.step()

    if verbose>0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()


def evalModel(data_loader, model, device, verbose=0, stochastic_pass = True,
                 compute_metrics=True, activationName = None,
                 loss_fn = 'crossEnt'):

    if stochastic_pass:
        model.train()
    else:
        model.eval()

    test_loss = 0
    predictions = []
    activations = []
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)
                
            output = model(data)

            if compute_metrics:
                predictionClasses = output.argmax(dim=1, keepdim=True)

                if loss_fn == 'crossEnt':
                    criteria = nn.CrossEntropyLoss().cuda()
                    correct += predictionClasses.eq(target.view_as(predictionClasses)).sum().item()
                elif loss_fn == 'bceLogit':
                    criteria = nn.BCEWithLogitsLoss().cuda()
                    correct += predictionClasses.eq(target.argmax(dim=1).view_as(predictionClasses)).sum().item()

                test_loss += criteria(output, target).sum().item()
            else:
                softmaxed = F.softmax(output.cpu(), dim=1)
                predictions.extend(softmaxed.data.numpy())

    if compute_metrics:
        return test_loss, correct
    else:
        return predictions, activations
    

def test(model, test_loader, device, verbose=0, activationName = None,
            loss_fn = 'crossEnt'):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0

    total_test_loss, total_corrections = evalModel(test_loader, model, device=device, 
                                                    verbose = verbose, 
                                                    stochastic_pass = False, compute_metrics = True, 
                                                    activationName = activationName,
                                                    loss_fn=loss_fn)

    test_loss = total_test_loss/ len(test_loader) # loss function already averages over batch size
    test_acc = total_corrections / len(test_loader.dataset)
    if verbose>0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        print('{{"metric": "Eval - cross entropy Loss", "value": {}, "epoch": {}}}'.format(
            test_loss, epoch))
        print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
            100. * correct / len(test_loader.dataset), epoch))

    return test_loss, test_acc