# Spectral Synthesis for Satellite-to-Satellite Translation

## Model

VAE-GAN Architecture forr unsupervised image-to-image translation with shared spectral reconstruction loss. Model is trained on GOES-16/17 and Himawari-8 L1B data processed by GeoNEX. 

![Network Architecture](images/image-to-image-sensors.png)

![Alt Text](synthetic_animation.gif)

## Data

https://drive.google.com/file/d/1gzLcqWiKPjvzltp2nVZH6uCSIQ0G2h1u/view?usp=sharing

## Dependencies

Python==3.7 <br>
Pytorch==1.5 <br>
Petastorm==0.9

Note: Functionality using PyTorch with MPI requires installation from source.

```
conda create --name geonex_torch1.5 python=3.7 pytorch=1.5 xarray numpy scipy pandas torchvision tensorboard opencv pyyaml jupyterlab matplotlib seaborn
conda install -c conda-forge pyhdf
pip install petastorm
```

## Steps to Reproduce Experiments

To Come

## Acknowledgements 

We acknowledge the network codes inherented from https://github.com/mingyuliutw/UNIT
