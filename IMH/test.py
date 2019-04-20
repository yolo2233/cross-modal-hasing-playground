import scipy.io as sio
import numpy as np
ham_dis = sio.loadmat('temp_data/ham_dis.mat')
distance_i2t = ham_dis['ham_dis_i2t']
distance_t2i = ham_dis['ham_dis_t2i']
print(distance_i2t.shape)
print(distance_t2i.shape)
print(np.max(distance_i2t))
print(np.max(distance_t2i))