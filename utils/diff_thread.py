import numpy as np
import matplotlib.pyplot as plt

import argparse

def fijii_np(path,shape,type='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

finalOuterIter = 5000
MSE_normed = np.zeros((finalOuterIter,1))
root = '/home/meraslia/workspace_reco/nested_admm/data/Algo/image2_0/replicate_1/'
subfolder1 = 'ADMMLim_test_1_frame1_float'
subfolder2 = 'ADMMLim_test_48_frame_1_float'
PETImage_shape = (112,112,1)

for outer_it in range(1,finalOuterIter+1):
    img1_np = fijii_np(root + subfolder1 + '/' + subfolder1 + '_it' + str(outer_it) + ".img", shape=(PETImage_shape))
    img2_np = fijii_np(root + subfolder2 + '/' + subfolder2 + '_it' + str(outer_it) + ".img", shape=(PETImage_shape))

    MSE_normed[outer_it - 1] = np.linalg.norm(img1_np - img2_np) / (PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])
    print("outer_it : ",outer_it)
    print("MSE : ",MSE_normed[outer_it - 1])
    #print("Numerical error below threshold : ",MSE_normed < 1e-5)

fig, ax1 = plt.subplots()
plt.plot(np.arange(1,finalOuterIter+1),MSE_normed)
plt.title("MSE between 1 thread and 48 threads")
plt.xlabel("it")
plt.ylabel("MSE")
#ax1.set_ylim(-2,3)
plt.savefig("MSE between 1 thread and 48 threads")
