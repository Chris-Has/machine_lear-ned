---
toc: true
layout: post
description: A tutorial on how to install and run the CPU version of fastai
categories: [markdown]
title: Installing fastai (CPU only) on windows 10
---
# Installing fastai (CPU only) on windows 10

<b>Step 1:</b> Install Anaconda\
&emsp; Default settings are fine\
<b>Step 2:</b> Create conda environment\
&emsp; ```conda create -n fastai```\
<b>Step 3:</b> Activate conda environment\
&emsp; ```activate fastai```\
<b>Step 4:</b> Install fastai and pytorch\
&emsp;  ```conda install -c fastai -c pytorch fastai```\
<b>Step 5:</b> Install jupyter\
&emsp; ```conda install jupyter```\
<b>Step 5:</b> Test installation\
&emsp; Use example from first lesson
```
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #Tricks pytorch into thinking a gpu is not available - run each time
from fastai.vision.all import *

defaults.device = torch.device('cpu')
defaults.device
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper();
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224),
    device='cpu',
    num_workers=0)

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```
