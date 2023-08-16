import numpy as np

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

PETImage_shape = (112,112)
target = np.zeros((PETImage_shape))

nb_rois = 1

if (nb_rois == 1):
    target[:,:] = 50
if (nb_rois == 2):
    target[:,0:int(PETImage_shape[0]/nb_rois)] = 10
    target[:,int(PETImage_shape[0]/nb_rois):int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois):] = 100
elif (nb_rois == 3):
    target[:,0:int(PETImage_shape[0]/nb_rois)] = 10
    target[:,int(PETImage_shape[0]/nb_rois):int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois):] = 50
    target[:,int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois):] = 100
elif (nb_rois == 4):
    target[:,0:int(PETImage_shape[0]/nb_rois)] = 0
    target[:,int(PETImage_shape[0]/nb_rois):int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois):] = 10
    target[:,int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois):int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois):] = 20
    target[:,int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois)+int(PETImage_shape[0]/nb_rois):] = 100

import matplotlib.pyplot as plt
plt.imshow(target,cmap="gray")
plt.colorbar()
plt.show()

save_img(target,"/disk/workspace_reco/nested_admm/data/Algo/Data/relu_exp_target_" + str(nb_rois) + "_rois_50.img")



PETImage_shape = (3,3)
target = np.ones((PETImage_shape))
save_img(target,"data/Algo/Data/database_v2/image_relu_exp/image_relu_exp.raw")