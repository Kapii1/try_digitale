import tensorflow as tf
import os
import numpy as np
somme=0
l=np.load('features2.npy', mmap_mode='r')
for i in l:
    somme+= len(i)
print(l)