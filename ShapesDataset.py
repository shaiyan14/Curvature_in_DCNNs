# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:39:30 2019

@author: shaiy
"""
import pandas as pd
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import time


class ShapesDataset(Dataset):

    def __init__(self, creature, root_dir, class_tuples=None, transform=None):
        """
        Args:
            creature (string): name of the creature
            root_dir (string): should be "shapes".
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.creature = creature
        self.root_dir = root_dir
        self.transform = transform
        self.class_tuples = class_tuples

    def __len__(self):
        class_tuples = self.class_tuples

        if isinstance(class_tuples,type(None)):
            path, dirs, files = next(os.walk(os.path.join(self.root_dir,self.creature)))
            return len(files)
        else:
            return len(class_tuples)


    def __getitem__(self, idx):
        class_tuples = self.class_tuples

        if torch.is_tensor(idx):
            idx = idx.tolist()


        if isinstance(class_tuples,type(None)):
            shape_dir = os.path.join(self.root_dir,self.creature, self.creature + '_' +
                                     str(idx) + '.csv')
        else:
            shape_dir = class_tuples[idx][0]

        points = pd.read_csv(shape_dir,header=None)

        if self.transform:
            image = self.transform(points)

        sample = {'image': image, 'label': np.double(class_tuples[idx][1])}

        return sample

class ZeroMeanPoints(object):
    def __call__(self, points):
        points= points- np.mean(points,axis=1,keepdims=True)
        return points

class MatchPerimeterPoints(object):
    def __call__(self,points):
        x_i = points[0,:-1]
        x_i_plus_1 = points[0,1:]
        y_i = points[1,:-1]
        y_i_plus_1 = points[1,1:]

        segment_lengths = np.sqrt((y_i_plus_1-y_i)**2+(x_i_plus_1-x_i)**2)
        perimeter = np.sum(segment_lengths)

        points = (3.5) * points/(perimeter)

        return points

class MatchAreaPoints(object):
    def __call__(self,points):
        x_i = points[0,:-1]
        x_i_plus_1 = points[0,1:]
        y_i = points[1,:-1]
        y_i_plus_1 = points[1,1:]

        dataset_area = np.abs(0.5*np.sum(x_i*y_i_plus_1 - y_i*x_i_plus_1))

        points = points/np.sqrt(dataset_area)
        points = points/3
        return points

class RandomRotatePoints(object):

    '''
        single input means +/-, None means random, tuple defines range

    '''

    def __init__(self, rot_angle=None):
        assert isinstance(rot_angle, (int, float, type(None), tuple))
        self.rot_angle = rot_angle

    def __call__(self, points):
        rot_angle = self.rot_angle

        if isinstance(rot_angle,(int,float)):
            rot_angle_sample = np.random.uniform(low=-rot_angle,high=rot_angle)

        elif isinstance(rot_angle,tuple):
            rot_angle_sample = np.random.uniform(rot_angle[0],rot_angle[1])

        elif isinstance(rot_angle,type(None)):
            rot_angle_sample = np.random.uniform(0,360)

        rot_angle_sample_rads = rot_angle_sample*(np.pi/180.)


        rot_matrix = np.array([[np.cos(rot_angle_sample_rads),-np.sin(rot_angle_sample_rads)],[np.sin(rot_angle_sample_rads),np.cos(rot_angle_sample_rads)]])
        points = rot_matrix.dot(points)

        return points

class GetFragmentPoints(object):
    def __init__(self, start_position=None, length=None):
        assert isinstance(start_position, (int, type(None)))
        assert isinstance(length, (int, type(None)))
        self.start_position = start_position
        self.length = length

    def __call__(self, points):

        start_position = self.start_position
        length = self.length

        doubled_dataset = np.concatenate((points,points),axis=1)

        if isinstance(length,type(None)):
            length = points.shape[1]

        if isinstance(start_position,int):
            points = doubled_dataset[:,start_position:(start_position+length+1)]
        else:
            tmp = np.random.randint(0,points.shape[1])
            points = doubled_dataset[:,tmp:(tmp+length+1)]

        return points

class RasterPoints(object):
    def __init__(self, image_dim = 224, fill = False, AA_multiplier= 5):
        assert isinstance(image_dim, int)
        assert isinstance(fill, bool)

        self.image_dim = image_dim
        self.fill = fill
        self.AA_multiplier = AA_multiplier
    def __call__(self, points):
        image_dim = self.image_dim
        fill = self.fill

        # for antialiasing
        AA_multiplier = self.AA_multiplier
        image_dim = AA_multiplier*image_dim

        points = points*image_dim/2+image_dim/2
        image = Image.new("L", (image_dim, image_dim),color=255)

        draw = ImageDraw.Draw(image)
        points = tuple(map(tuple, np.transpose(points,axes=[1,0])))
        draw.line((points), fill= 0, width=AA_multiplier)
        if fill:
            draw.polygon((points), fill= 0)
        image = image.rotate(180)

        image_dim = int(image_dim/AA_multiplier)
        image_new = image.resize((image_dim,image_dim),resample=Image.BILINEAR)
        return np.asarray(image_new)

class CSVToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, csv_points):
        points = np.array(csv_points)
        return points

class RepMatImage(object):
    def __call__(self, image):

        image = np.tile(image,(3,1,1))

        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = (np.expand_dims(image,axis=0)).astype(float)
        return torch.from_numpy(image.astype(float)).float()

class Normalize(object):
    """Set the mean and std of the image."""
    def __init__(self, new_mean=[0, 0, 0], new_std=[1, 1, 1]):
        assert isinstance(new_mean, list)
        assert isinstance(new_std, list)

        self.new_mean = np.asarray(new_mean)
        self.new_std = np.asarray(new_std)

    def __call__(self, image):
        new_image = np.zeros(image.shape)
        for dim in range(0,image.shape[0]):
            zero_meaned = image[dim,:,:] - np.mean(image[dim,:,:])
            std_matched = self.new_std[dim]*zero_meaned/np.std(zero_meaned)
            new_image[dim,:,:] = std_matched + self.new_mean[dim]

        return new_image

class ShowImage(object):
    """Convert from tensor to PIL image and draw"""

    def __call__(self, tensor):
        tmp  = np.transpose(np.asarray(tensor),axes=(1,2,0))
        image = Image.fromarray(np.uint8(5*tmp+128.))
        image.show()
        crash
