# Curvature_in_DCNNs

This project uses linear decoding of intermediate DCNN activations to elucidate the representation of non-local shape information in DCNNs. Note that the word "animals" is used in this project because the shapes are of animals. This will likely be changed in the future. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

This project uses PyTorch and torchvision in Python. This project uses miniconda for package management ([environment.yml](environment.yml)). Currently the parameters are set to use CUDA on an NVIDIA GPU, but this can be changed as a passed parameter when initializing a `Curve_comparison` object. 

## Usage

1. Initialize a `Curve_comparison` object from [animals_experiment_functions.py](animals_experiment_functions.py).
2. Use `torchvision.transforms.Compose` with a [ShapesDataset](ShapesDataset.py) dataset to create training and testing transforms.
3. Set the fragment length (if using shape fragments) with `animals_experiment_functions.set_fragment_length(fragment_length: int)`.
4. Run the analysis. Use `animals_experiment_functions.CV_performance_over_feature_layers` to get cross validated performance or edit the example in [`run_animal_classification_experiment_CSV.py`](run_animal_classification_experiment_CSV.py)

## License

MIT License