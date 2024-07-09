import numpy as np
import matplotlib.pyplot as plt

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""               
    file_path=(path)
    if (1 in shape):
        nb_dimensions = 2
    else:    
        nb_dimensions = 3
    dtype_np = np.dtype(type_im)
    with open(file_path, 'rb') as fid:
        data = np.fromfile(fid,dtype_np)
        if (nb_dimensions == 2): # 2D
            image = data.reshape(shape)
        else: # 3D
            image = data.reshape(shape[::-1])
    return image

finalOuterIter = 5000
MSE_normed = np.zeros((finalOuterIter,1))
subroot = "data/Algo/"
subsubroot = 'image2_0/replicate_1/'
subfolder1 = 'ADMMReg_test_1_frame1_float'
subfolder2 = 'ADMMReg_test_48_frame_1_float'
PETImage_shape = (112,112,1)

for inner_it in range(1,finalOuterIter+1):
    img1_np = fijii_np(subroot + subsubroot + subfolder1 + '/' + subfolder1 + '_it' + str(inner_it) + ".img", shape=(PETImage_shape))
    img2_np = fijii_np(subroot + subsubroot + subfolder2 + '/' + subfolder2 + '_it' + str(inner_it) + ".img", shape=(PETImage_shape))

    MSE_normed[inner_it - 1] = np.linalg.norm(img1_np - img2_np) / (PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])
    print("inner_it : ",inner_it)
    print("MSE : ",MSE_normed[inner_it - 1])
    #print("Numerical error below threshold : ",MSE_normed < 1e-5)

fig, ax1 = plt.subplots()
plt.plot(np.arange(1,finalOuterIter+1),MSE_normed)
plt.title("MSE between 1 thread and 48 threads")
plt.xlabel("it")
plt.ylabel("MSE")
#ax1.set_ylim(-2,3)
plt.savefig("MSE between 1 thread and 48 threads")
