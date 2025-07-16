import numpy as np
from importlib import reload

import torch

import torchvision.models as tmodels
from torchvision import transforms
import animals_experiment_functions as af
import ShapesDataset as sd

"""
Example code to run a shape information analysis with "animals_experiment_functions.py" 
"""

# for reproducibility and speed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# number of CV splits
k = 5

# classes to compare
fragments_list = np.round(np.linspace(2,120,7)).astype(int)
class_0_list=[]
class_1_list=[]
class_0 = 'animals'
class_1 = 'sparse_animals'
for ii in fragments_list:
    class_0_list.append(class_0 + '_' + str(ii))
    class_1_list.append(class_1 + '_' + str(ii))

results = dict()
model_name = 'alexnet'

# model comparison object
model_comparison = af.Curve_comparison(model_name,
         class_0,
         class_1,
         '',
         '',
         epochs= 5,
         k= k,
         random_weights= False,
         save_weights= False,
         load_weights= False,
         save_results= True,
         use_gpu= True,
         run_name= 'testing',
         image_folder= '',
         task_2AFC= True,
         csv_folder= 'shapes')

# do a binary comparison between animals ('animals') and sparse components of animal shape ('sparse_animals')
for fragment in fragments_list:

    # transforms for training and validation
    training_transform = transforms.Compose([
        # you can add other transformations in this list
        sd.CSVToNumpy(),
        sd.MatchPerimeterPoints(),
        sd.GetFragmentPoints(length=int(fragment)),
        sd.RandomRotatePoints((-180,180)),
        sd.ZeroMeanPoints(),
        sd.RasterPoints(fill= False),
        sd.RepMatImage(),
        sd.Normalize(new_mean=[0.456, 0.456, 0.456], new_std=[0.225,  0.225, 0.225]),
        #sd.ShowImage(),
        sd.ToTensor()
    ])

    model_comparison.training_transform = training_transform

    valid_transform = transforms.Compose([
    # you can add other transformations in this list
        sd.CSVToNumpy(),
        sd.MatchPerimeterPoints(),
        sd.GetFragmentPoints(length=int(fragment)),
        sd.RandomRotatePoints((-180,180)),
        sd.ZeroMeanPoints(),
        sd.RasterPoints(fill= False),
        sd.RepMatImage(),
        sd.Normalize(new_mean=[0.456, 0.456, 0.456], new_std=[0.225,  0.225, 0.225]),
        sd.ToTensor()
    ])

    model_comparison.validation_transform = valid_transform
    model_comparison.set_fragment_length(int(fragment))

    [pc_all_feat,hr_all_feat,fa_all_feat] = model_comparison.CV_performance_over_feature_layers()

    results[str(fragment)] = pc_all_feat

    model_comparison.save_properties_function()
