
# coding: utf-8

# In[1]:


from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, HTML
from tqdm import trange
import time
from datetime import datetime
from preprocess import load_data, preprocess_data, apply_projection_transform, visualize_dataset, images_show, visualize_dataset, summarize_histogram,ahisteq

import pandas as pd

from skimage import img_as_ubyte, img_as_float
import cv2

import numpy as np
import math
from random import randint
from collections import namedtuple

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


train_set = 'train.pickle'
valid_set = 'valid.pickle'
test_set = 'test.pickle'

summarize_histogram(train_set, test_set, valid_set)
X_train, y_train = load_data(train_set)
visualize_dataset(X_train, y_train, view_histogram=True, show_images=True, show_all_classes=True)
X_test,Y_test=load_data(test_set)
# visualize_dataset(X_test,Y_test,view_histogram=True, show_images=True, show_all_classes=True)
X_valid,Y_valid=load_data(valid_set)
# visualize_dataset(X_valid,Y_valid,view_histogram=True, show_images=True, show_all_classes=True)

