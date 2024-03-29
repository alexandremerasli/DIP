import numpy as np
from pathlib import Path

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

nb_frames = 10
it = 200
im_3D = np.zeros((nb_frames,112,112))
root = "/home/meraslia/workspace_reco/test_dynamic_2_type_frame/"

for frame in range(1,nb_frames+1):
    #filename = root + "framebyframe64/FrbyFr_rep1_it" + str(it) + "_fr" + str(frame)
    subfolder = "framebyframe_all_norm_test"
    filename = root + subfolder + '/' + subfolder + "_it" + str(it) + "_fr" + str(frame)
    path = Path(filename)
    print(path)
    slice = fijii_np(filename + ".img",(112,112))
    im_3D[frame-1,:,:] = slice

save_img(im_3D,root + subfolder + "it_" + str(it) + "_stacked.img")