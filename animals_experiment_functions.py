# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:11:47 2019

@author: Shaiyan Keshvari
"""

import numpy as np
from scipy.stats import norm

import torch
import torch.nn as nn

import torchvision.models as tmodels
import torch.optim as optim
import torchvision

import copy

from PIL import Image

import h5py

import os, os.path
import glob
from datetime import datetime
import pickle
import pandas as pd

import matplotlib.pyplot as plt

import ShapesDataset as sd

class Curve_comparison:

    def __init__(self,
                 model_name,
                 class_0,
                 class_1,
                 training_transform,
                 validation_transform,
                 epochs=5,
                 k=5,
                 random_weights=False,
                 save_weights=False,
                 load_weights=False,
                 save_results=True,
                 run_name= '',
                 use_gpu=True,
                 image_folder = 'Curvature_all',
                 task_2AFC=True,
                 csv_folder = 'shapes',
                 fragment_length = 120):

        if load_weights:
            pickle_off = open('./saved_linear_weights/' + run_name + '/model_instance.pkl',"rb")
            tmp_object = pickle.load(pickle_off)
            self.model_name = tmp_object.model_name
            self.layers_to_try = tmp_object.layers_to_try
            self.class_0 = tmp_object.class_0
            self.class_1 = tmp_object.class_1
            self.training_transform = tmp_object.training_transform
            self.validation_transform = tmp_object.validation_transform
            self.epochs = tmp_object.epochs
            self.k = tmp_object.k
            self.random_weights = tmp_object.random_weights
            self.datestr = tmp_object.datestr
            self.save_results = tmp_object.save_results
            self.image_folder = tmp_object.image_folder
            self.task_2AFC = tmp_object.task_2AFC
            self.csv_folder = tmp_object.csv_folder
            self.fragment_length = tmp_object.fragment_length
        else:
            self.model_name = model_name
            self.class_0 = class_0
            self.class_1 = class_1
            self.training_transform = training_transform
            self.validation_transform = validation_transform
            self.epochs = epochs
            self.k = k
            self.random_weights = random_weights
            self.datestr = datetime.now().strftime('%Y%m%dT%H%M%S')
            self.save_results = save_results
            self.image_folder = image_folder
            self.task_2AFC = task_2AFC
            self.csv_folder = csv_folder
            self.fragment_length = fragment_length
            # for saving things
            if random_weights:
                random_weights_txt = 'random'
            else:
                random_weights_txt = 'pretrained'
            run_name = model_name + '_' + random_weights_txt + '_' + self.datestr

        self.save_weights = save_weights
        self.use_gpu = use_gpu
        self.load_weights = load_weights
        self.run_name = run_name
        self.all_results = dict()

        # get the layers to try
        self.generate_partial_model(-1)
        layers_to_try = get_relu_indices(self.net)
        self.layers_to_try = layers_to_try

        if save_weights:
            os.makedirs('./saved_linear_weights/' + run_name + '/',exist_ok=True)
            pickling_on = open('./saved_linear_weights/' + run_name + '/model_instance.pkl',"wb")
            pickle.dump(self, pickling_on)
            pickling_on.close()

    def set_class(self,new_class,class_num):
        if class_num == 0:
            self.class_0 = new_class
        elif class_num == 1:
            self.class_1 = new_class

    def set_fragment_length(self,new_fragment_length):
        self.fragment_length = new_fragment_length

    def generate_partial_model(self,last_layer):

        model_name = self.model_name
        use_gpu = self.use_gpu
        random_weights = self.random_weights
        task_2AFC = self.task_2AFC

        if model_name == 'alexnet_places365':
            arch = 'alexnet'

            # load the pre-trained weights
            model_file = 'other_networks/%s_places365.pth.tar' % arch

            model = tmodels.__dict__[arch](num_classes=365)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            net = model
        else:
            if random_weights:
                net = eval('tmodels.' + model_name + '(pretrained=False)')
            else:
                net = eval('tmodels.' + model_name + "(weights='DEFAULT')")


        net = sequentialize_model(net,model_name)
        if last_layer == -1:
            last_layer = len(net)

        # don't collect gradients on the network
        for param in net.parameters():
            param.requires_grad = False

        # cut the network off at the ReLu
        net = net[0:last_layer+1]
        b = net(torch.FloatTensor(np.random.randn(1,3,224,224)))
        num_regressors = b.data.numel()
        net.add_module('flattener',Flatten())
        
        if not task_2AFC:
            net.add_module('final_layer',torch.nn.Linear(num_regressors,2))

        if use_gpu:
            net = net.cuda()

        self.net = net
        self.num_regressors = num_regressors

    def train_model(self):

        epochs = self.epochs
        net = self.net
        num_regressors = self.num_regressors
        use_gpu = self.use_gpu
        task_2AFC = self.task_2AFC

        if task_2AFC:
            # class 0 will always be the positive decision variable
            data_loader_train_0 = self.data_loader_train_0
            data_loader_train_1 = self.data_loader_train_1
            TwoAFC_layer = TwoAFC(num_regressors)

            if use_gpu:
                TwoAFC_layer = TwoAFC_layer.cuda()
            optimizer = optim.SGD(TwoAFC_layer.parameters(), lr=0.01, momentum=0.9)
        else:
            data_loader = self.data_loader_train
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        print('training: ', end='')
        for epoch in range(epochs):  # loop over the dataset multiple times
            print(str(100.*(epoch+1)/epochs)[0:4] + '%, ', end='')
            if task_2AFC:
                for i, (data_0, data_1) in enumerate(zip(data_loader_train_0,data_loader_train_1), 0):
                    # get the inputs
                    inputs_0, labels_0 = data_0['image'], data_0['label']
                    inputs_1, labels_1 = data_1['image'], data_1['label']

                    if next(net.parameters()).is_cuda:
                        inputs_0 = inputs_0.cuda()
                        labels_0 = labels_0.cuda()
                        inputs_1 = inputs_1.cuda()
                        labels_1 = labels_1.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    loss = -TwoAFC_layer(net(inputs_0),net(inputs_1))
                    loss.backward()
                    optimizer.step()
            else:
                for i, data in enumerate(data_loader, 0):
                    # get the inputs
                    inputs, labels = data
                    if next(net.parameters()).is_cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()


        print('Finished Training')
        self.net = net
        if task_2AFC:
            self.TwoAFC_layer = TwoAFC_layer

    def evaluate_model(self):

        task_2AFC = self.task_2AFC
        net = self.net

        if task_2AFC:
            data_loader_valid_0 = self.data_loader_valid_0
            data_loader_valid_1 = self.data_loader_valid_1
            TwoAFC_layer = self.TwoAFC_layer
        else:
            data_loader = self.data_loader_valid

        # put net in eval mode so dropout disabled
        net.eval()
        if task_2AFC:
            predicted_label = np.zeros(len(data_loader_valid_0))
            actual_label = np.zeros(len(data_loader_valid_0))
            for i, (data_0, data_1) in enumerate(zip(data_loader_valid_0,data_loader_valid_1), 0):
                # get the inputs
                inputs_0 = data_0['image']
                inputs_1 = data_1['image']


                if next(net.parameters()).is_cuda:
                    inputs_0 = inputs_0.cuda()
                    inputs_1 = inputs_1.cuda()

                stim_order = np.random.randint(2)
                if stim_order == 0:
                    outputs = -TwoAFC_layer(net(inputs_0),net(inputs_1))
                else:
                    outputs = -TwoAFC_layer(net(inputs_1),net(inputs_0))

                if outputs.detach().cpu().numpy() < 0:
                    predicted_label[i] = 0
                else:
                    predicted_label[i] = 1

                actual_label[i] = stim_order
        else:
            predicted_label = np.zeros(len(data_loader))
            actual_label = np.zeros(len(data_loader))

            # get network predictions for all the data
            for i, data in enumerate(data_loader, 0):
                # get the inputs
                inputs, labels = data

                if next(net.parameters()).is_cuda:
                    inputs = inputs.cuda()

                outputs = net(inputs)

                predicted_label[i] = np.argmax(outputs.detach().cpu().numpy())
                actual_label[i] = labels.detach().cpu().numpy()

        # return percent correct, return hits and false alarms, assuming class_0 is the signal

        pc = (np.mean(predicted_label == actual_label))
        hr = np.sum(np.logical_and(actual_label == 0, predicted_label == 0))/np.sum(actual_label == 0)
        fa = np.sum(np.logical_and(actual_label == 1, predicted_label == 0))/np.sum(actual_label == 1)
        print('Accuracy: ' + f"{pc:.4f}")

        return pc, hr, fa


    def collate_model_responses_into_numpy_array(self):

        # cut out the last linear layer
        data_loader = self.data_loader
        old_net = self.net

        net = copy.deepcopy(old_net)
        feat_length = old_net.final_layer.in_features
        net.classifier = net[:-1]

        net.eval()
        all_outputs = np.zeros((len(data_loader),feat_length))
        all_labels =  np.zeros((len(data_loader)))
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            inputs, labels = data
            if next(net.parameters()).is_cuda:
                inputs = inputs.cuda()

            outputs = net(inputs)

            all_outputs[i,:] = outputs.detach().cpu().numpy()
            all_labels[i] = labels.detach().cpu().numpy()

        return all_outputs, all_labels

    def generate_train_test_split_tuple_lists(self):

        class_0 = self.class_0
        class_1 = self.class_1
        k = self.k
        image_folder = self.image_folder

        class_0_images = glob.glob(image_folder + '\\' + class_0 +'_images\\\\*.png')
        class_1_images = glob.glob(image_folder + '\\' + class_1 +'_images\\\\*.png')

        # keep the number of samples corresponding to the smaller database
        samples_to_keep = min(len(class_0_images),len(class_1_images))
        samples_per_split = float(samples_to_keep)/float(k)
        splits = []
        for i in range(0,k):
            splits.append(list(range(int(i*samples_per_split),int((i+1)*samples_per_split))))

        np.random.shuffle(class_0_images)
        np.random.shuffle(class_1_images)

        class_0_tuples = []
        class_1_tuples = []
        for curr_img in class_0_images:
            class_0_tuples.append((curr_img,0))

        for curr_img in class_1_images:
            class_1_tuples.append((curr_img,1))

        self.class_0_tuples = class_0_tuples
        self.class_1_tuples = class_1_tuples
        self.splits = splits

    def generate_train_test_split_tuple_lists_CSV(self):

        class_0 = self.class_0
        class_1 = self.class_1
        csv_folder = self.csv_folder
        k = self.k

        class_0_points = glob.glob(os.path.join(csv_folder,class_0,class_0 + '*.csv'))
        class_1_points = glob.glob(os.path.join(csv_folder,class_1,class_1 + '*.csv'))

        # keep the number of samples corresponding to the smaller database
        samples_to_keep = min(len(class_0_points),len(class_1_points))
        samples_per_split = float(samples_to_keep)/float(k)
        splits = []
        for i in range(0,k):
            splits.append(list(range(int(i*samples_per_split),int((i+1)*samples_per_split))))

        np.random.shuffle(class_0_points)
        np.random.shuffle(class_1_points)

        class_0_tuples = []
        class_1_tuples = []
        for curr_point in class_0_points:
            class_0_tuples.append((curr_point,0))

        for curr_point in class_1_points:
            class_1_tuples.append((curr_point,1))

        self.class_0_tuples = class_0_tuples
        self.class_1_tuples = class_1_tuples
        self.splits = splits

    def generate_train_test_dataloaders(self):

        class_0_tuples = self.class_0_tuples
        class_1_tuples = self.class_1_tuples
        splits = self.splits
        curr_rep = self.curr_rep
        training_transform = self.training_transform
        validation_transform = self.validation_transform
        k = self.k

        train_idx = list(set(range(0,k))-set([curr_rep]))

        train_image_indices = []
        for i in train_idx:
            #print(i)
            train_image_indices+=splits[i]

        val_image_indices = splits[curr_rep]

        # tiny imagenet training images
        train = torchvision.datasets.ImageFolder('tmp_dont_delete',transform=training_transform)
        train.imgs.pop()
        valid = torchvision.datasets.ImageFolder('tmp_dont_delete',transform=validation_transform)
        valid.imgs.pop()

        for index in train_image_indices:
            train.imgs.append(class_0_tuples[index])
            train.imgs.append(class_1_tuples[index])
        for index in val_image_indices:
            valid.imgs.append(class_0_tuples[index])
            valid.imgs.append(class_1_tuples[index])

        data_loader_valid = torch.utils.data.DataLoader(valid,shuffle=False)
        data_loader_train = torch.utils.data.DataLoader(train,shuffle=True)

        self.data_loader_valid = data_loader_valid
        self.data_loader_train = data_loader_train

    def generate_dataloaders_CSV(self):

        class_0_tuples = self.class_0_tuples
        class_1_tuples = self.class_1_tuples
        class_0 = self.class_0
        class_1 = self.class_1
        splits = self.splits
        curr_rep = self.curr_rep
        training_transform = self.training_transform
        validation_transform = self.validation_transform
        k = self.k
        csv_folder = self.csv_folder
        train_idx = list(set(range(0,k))-set([curr_rep]))

        train_image_indices = []
        for i in train_idx:
            train_image_indices+=splits[i]

        val_image_indices = splits[curr_rep]

        # training and validation images
        train_class_0 = sd.ShapesDataset(class_tuples=[class_0_tuples[ii] for ii in train_image_indices],creature=class_0, root_dir=csv_folder, transform=training_transform)
        valid_class_0 = sd.ShapesDataset(class_tuples=[class_0_tuples[ii] for ii in val_image_indices], creature=class_0, root_dir=csv_folder, transform=validation_transform)

        train_class_1 = sd.ShapesDataset(class_tuples=[class_1_tuples[ii] for ii in train_image_indices],creature=class_1, root_dir=csv_folder, transform=training_transform)
        valid_class_1 = sd.ShapesDataset(class_tuples=[class_1_tuples[ii] for ii in val_image_indices], creature=class_1, root_dir=csv_folder, transform=validation_transform)

        data_loader_valid_0 = torch.utils.data.DataLoader(valid_class_0,batch_size=1,shuffle=False)
        data_loader_train_0 = torch.utils.data.DataLoader(train_class_0,batch_size=32,shuffle=True)

        data_loader_valid_1 = torch.utils.data.DataLoader(valid_class_1,batch_size=1,shuffle=False)
        data_loader_train_1 = torch.utils.data.DataLoader(train_class_1,batch_size=32,shuffle=True)

        self.data_loader_valid_0 = data_loader_valid_0
        self.data_loader_train_0 = data_loader_train_0
        self.data_loader_valid_1 = data_loader_valid_1
        self.data_loader_train_1 = data_loader_train_1

    def CV_performance_over_feature_layers(self):

        layers_to_try = self.layers_to_try
        class_0 = self.class_0
        class_1 = self.class_1
        save_weights = self.save_weights
        load_weights = self.load_weights
        k = self.k
        run_name = self.run_name
        save_results = self.save_results
        task_2AFC = self.task_2AFC

        pc_all = np.zeros((k,len(layers_to_try)))
        hr_all = np.zeros((k,len(layers_to_try)))
        fa_all = np.zeros((k,len(layers_to_try)))

        # use 5 k split for validation
        self.generate_train_test_split_tuple_lists_CSV()

        print('class 0: ' + class_0 + ', class 1: ' + class_1)
        print('Fragment length ' + str(self.fragment_length))

        for curr_rep in range(0,k):

            print('\nCV rep ' + str(curr_rep+1) + ' of ' + str(k))
            self.curr_rep = curr_rep

            if task_2AFC:
                self.generate_dataloaders_CSV()
            else:
                self.generate_train_test_dataloaders()


            index = 0
            regress_list= []

            for layer in layers_to_try:
                self.generate_partial_model(layer)
                net = self.net

                print('layer ' + str(index+1) + ' of ' + str(len(layers_to_try)) )

                filename = './saved_linear_weights/' + run_name + '/' + class_0 + '_vs_' +  class_1 + '_layer_' +  '_CVrep_' + str(curr_rep) + ".pkl"

                if load_weights:
                    print('Loading weights...')

                    net[-1].load_state_dict(torch.load(filename))
                else:
                    self.train_model()

                if save_weights:
                    print('Saving weights...')
                    torch.save(net[-1].cpu().state_dict(),filename)

                #assuming you want to use gpu
                if self.use_gpu:
                    net.cuda()
                else:
                    net.cpu()

                self.net = net

                pc, hr, fa = self.evaluate_model()

                pc_all[curr_rep,index]=pc
                hr_all[curr_rep,index]=hr
                fa_all[curr_rep,index]=fa

                regress_list.append(self.num_regressors)
                index += 1

        self.pc_all = pc_all
        self.hr_all = hr_all
        self.fa_all = fa_all

        if save_results:
            self.save_results_function()

        return pc_all, hr_all, fa_all

    def MI_feature_layers(self):

        layers_to_try = self.layers_to_try
        class_0 = self.class_0
        class_1 = self.class_1
        run_name = self.run_name
        save_results = self.save_results
        self.curr_rep = 0
        self.generate_train_test_split_tuple_lists_CSV()
        self.generate_dataloaders_CSV()
        data_loader_train_0 = self.data_loader_train_0
        data_loader_train_1 = self.data_loader_train_1
        data_loader_valid_0 = self.data_loader_valid_0
        data_loader_valid_1 = self.data_loader_valid_1
        all_mi_per_layer = []
        layer_index = 0

        for layer in layers_to_try:
            self.generate_partial_model(layer)

            # can't fit both tensors in GPU memory simultaneously
            all_data_0 = np.zeros((391, self.num_regressors))
            all_data_1 = np.zeros((391, self.num_regressors))
            mi=self.num_regressors*[None]
            index = 0
            print('layer ' + str(layer_index+1) + ' of ' + str(len(layers_to_try)) )

            for i, (data_0, data_1) in enumerate(zip(data_loader_train_0,data_loader_train_1), 0):
                all_data_0[index,:] = self.net(data_0['image'].cuda()).cpu().numpy()
                all_data_1[index,:] = self.net(data_1['image'].cuda()).cpu().numpy()
                index+= 1

            for i, (data_0, data_1) in enumerate(zip(data_loader_valid_0,data_loader_valid_1), 0):
                all_data_0[index,:] = self.net(data_0['image'].cuda()).cpu().numpy()
                all_data_1[index,:] = self.net(data_1['image'].cuda()).cpu().numpy()
                index+= 1

            # compute histograms
            min_val_1 = np.min(all_data_0)
            max_val_1 = np.max(all_data_0)
            min_val_2 = np.min(all_data_1)
            max_val_2 = np.max(all_data_1)

            min_all = np.min([min_val_1, min_val_2])
            max_all = np.max([max_val_1, max_val_2])

            num_bins = 10000
            bin_width = (max_all-min_all)/np.double(num_bins)

            for i in range(0,all_data_0.shape[1]):
                all_data_0_hist = 0.5*np.histogram(all_data_0[:,i], bins=num_bins, range=(min_all,max_all), density=True)[0]*bin_width
                all_data_1_hist = 0.5*np.histogram(all_data_1[:,i], bins=num_bins, range=(min_all,max_all), density=True)[0]*bin_width

                joint_dist = np.vstack((all_data_0_hist,all_data_1_hist))
                tmp_sum = np.log2(joint_dist/(np.sum(joint_dist,axis=0,keepdims=True)*0.5))
                tmp_sum[np.logical_not(np.isfinite(tmp_sum))] = np.nan
                mi[i]=(np.nansum(np.nansum(joint_dist*tmp_sum,axis=1),axis=0))
                #print(i)
                if i % 5000 == 0:
                    print(100*np.double(i)/np.double(all_data_0.shape[1]))

            all_mi_per_layer.append(mi)
            layer_index += 1
            self.pc_all = all_mi_per_layer

            self.save_results_function()


        return all_mi_per_layer

    def plot_w_errorbars(self,layers_to_try,pc_all=[],legend_label=''):

        if not pc_all.any():
            pc_all = self.pc_all
        x = list(range(0,pc_all.shape[1]))

        tmp = np.std(pc_all,axis=0)/np.sqrt(pc_all.shape[0])

        plt.errorbar(x,np.mean(pc_all,axis=0),yerr=tmp,label=legend_label)
        plt.ylim([0.5,1.])

    def plot_d_prime_w_errorbars(self,layers_to_try,hr_all=[],fa_all=[]):

        #if not hr_all.any():
        hr_all = self.hr_all
        #if not fa_all.any():
        fa_all = self.fa_all

        x = list(range(0,fa_all.shape[1]))
        d_prime = norm.ppf(hr_all) - norm.ppf(fa_all)

        print(norm.cdf(fa_all))
        tmp = np.std(d_prime,axis=0)/np.sqrt(d_prime.shape[0])

        plt.errorbar(x,np.mean(d_prime,axis=0),yerr=tmp)
        plt.savefig('test.png',format= 'png')

    def save_results_function(self):
        class_0 = self.class_0
        class_1 = self.class_1
        run_name = self.run_name
        pc_all = self.pc_all
        fragment_length = self.fragment_length
        all_results = self.all_results

        result = dict()
        result['pc_all'] = pc_all
        result['fragment_length'] = fragment_length
        result['class_0'] = class_0
        result['class_1'] = class_1
        all_results[fragment_length]=result

        os.makedirs('./results/' + run_name, exist_ok=True)
        pickling_on = open('./results/' + run_name + '/' + 'run_data.pkl',"wb")

        pickle.dump(all_results, pickling_on)
        self.all_results = all_results

        #pickling_on.close()

    def save_properties_function(self):
        run_name = self.run_name
        class_0 = self.class_0
        class_1 = self.class_1

        new_dict = dict(model_name = self.model_name, layers_to_try = self.layers_to_try,
                        class_0 = self.class_0, class_1 = self.class_1, epochs = self.epochs,
                        k = self.k, random_weights=self.random_weights, datestr = self.datestr,
                        image_folder = self.image_folder, task_2AFC = self.task_2AFC,
                        run_name = self.run_name, load_weights = self.load_weights)

        os.makedirs('./results/' + run_name,exist_ok=True)
        pickling_on = open('./results/' + run_name + '/' + class_0 + '_vs_' + class_1 + '_properties' +  '.pkl',"wb")
        pickle.dump(new_dict, pickling_on)
        pickling_on.close()

        with open('./results/' + run_name + '/' + class_0 + '_vs_' + class_1 + '_properties' +  '.txt', 'w') as f:
            print(new_dict, file=f)

def get_relu_indices(net):

    layers_to_try=[]

    for i, layer in enumerate(net,0):
        if isinstance(layer, nn.ReLU):
            layers_to_try.append(i)

    return layers_to_try

def find_datasize(path1,path2):
    path, dirs, files = next(os.walk(path1))
    len1= len(files)
    return len1

def load_results(dirname):
    #path, dirs, files = next(os.walk(dirname))
    #new_dict = dict()

    infile = open(dirname + '/run_data.pkl','rb')
    loaded_dict = pickle.load(infile)

    return loaded_dict

def load_properties(filename):

    infile = open(filename,'rb')
    loaded_dict = pickle.load(infile)
    infile.close()

    return loaded_dict

def sequentialize_model(net,model_name):

    # convert the features and classifier organization into one list
    modules = nn.ModuleList()
    for curr_module in net.features:
        modules.append(curr_module)

    if 'alexnet' in model_name:
        modules.append(nn.AdaptiveAvgPool2d((6, 6)))
    elif 'vgg' in model_name:
        modules.append(nn.AdaptiveAvgPool2d((7, 7)))
    modules.append(Flatten())
    for curr_module in net.classifier:
        modules.append(curr_module)

    net = nn.Sequential(*modules)
    return net

def plot_results(results_dict,filename,save_figure=True,fig_ax=None):

    sorted_key_list = sorted(results_dict.keys())

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    for fragment_length in sorted_key_list:
        x_axis = range(0,results_dict[fragment_length]['pc_all'].shape[1])
        mean_perf = np.mean(results_dict[fragment_length]['pc_all'],axis=0)
        stderr_perf = np.std(results_dict[fragment_length]['pc_all'],axis=0)/np.sqrt(results_dict[fragment_length]['pc_all'].shape[0])
        ax.errorbar(x_axis,mean_perf,yerr=stderr_perf,label=str(fragment_length))
        plt.ylim([0.4,1.])

    #ax.set_ylabel(results_dict[sorted_key_list[0]]['class_1'] + '\nPerformance')

    #plt.legend(results_dict.keys())

    if save_figure:
        fig.savefig(filename + '.svg',format= 'svg')
    plt.close()
    return fig, ax

def plot_averaged_results(csv_filename):

    rnames = pd.read_csv(csv_filename,header=None,dtype=str)
    for rname in rnames:
        results = load_results('results' + '/rname')


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

class TwoAFC(nn.Module):
    def __init__(self,nfeatures):
        super(TwoAFC,self).__init__()
        self.decoder = nn.Linear(nfeatures,1,bias=False)

    def forward(self,class_0,class_1):
        tmp = (self.decoder(class_0)-self.decoder(class_1))
        return(torch.sum(tmp)/len(tmp))
