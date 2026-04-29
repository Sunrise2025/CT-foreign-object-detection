# training_strategy

This package contains a set of tools to train and test classification neural networks.

The structure is inspired by msd_pytorch package.

Data loader includes standard augmentation techniques implemented without transforms from torchvision.

Model combines architectures from torchvision with a scaling layer for normalization.

Tensorboard is used for logging.

# Installation
```
conda create -n class_nn -c pytorch -c conda-forge 'pytorch=*=*cuda*' 'torchvision=*=*cu*' tensorboard 'imageio<2.15' scikit-learn scikit-image matplotlib numpy tqdm
pip install -e .
```

# Usage
The training script expects that a dataset folder consists of two subfolders trainining/ and validation/. Each should contain input/ subfolder with tiff images used as a network input and stats.csv file providing object information for every projection in the subset. It is not necessary to direcly assign classes to input images in the dataset. The package reads data from the stats.csv file and computes classes based on this information. For example, in the avocado dataset this file contains a number of voxels for peel, meat, pit and air. During the training process, you can define as many classes as you want based on the ratio between air and fruit volume. This way you can reuse the same data and try different classification problems.

The package saves tensorboard logs in the log/ subfolder and network parameters in the network_state/ subfolder.

```
python train.py --nn NN --data DATA --obj {playdoh,avocado} --run RUN
```

To start training, you need to specify the network architecture (currently, resnet50 or efficientnetb4), data folder (structured as explained above), object type (will be used to pick classification function), and run number. Run number is used to perform network training with different random seeds. Random seed can be included as optional argument. By default, a run number is used as a seed.

```
python apply.py --nn NN --data DATA --obj {playdoh,avocado}
```

This script is used for testing trained networks on the testing set. It will automatically look for all network runs that match dataset name specified as an input argument. The best epoch will be found automatically.

```
python plot_activation.py [-h] --nn NN --data DATA --obj {playdoh,avocado} --run RUN
```

This can be used to draw activation maps for the selected trained network. The implementation follows this code: https://github.com/joe3141592/PyTorch-CAM. 

