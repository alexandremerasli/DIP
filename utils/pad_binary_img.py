import numpy as np
from pathlib import Path

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

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

def write_hdr_img(path,filename):
    with open(path + ".hdr") as f:
        with open(path + "_cropped.hdr", "w") as f1:
            for line in f:
                if line.strip() == ('!matrix size [1] := 128'):
                    f1.write('!matrix size [1] := 112')
                    f1.write('\n')
                elif line.strip() == ('!matrix size [2] := 128'):
                    f1.write('!matrix size [2] := 112')
                    f1.write('\n')
                elif line.strip() == ('!name of data file := ' + filename + '.img'):
                    f1.write('!name of data file := ' + filename + '_cropped.img')
                    f1.write('\n')
                else:
                    f1.write(line)

# Root path
subroot = 'data/Algo/'

# Folder path from subroot where images to be padded are stored
subsubroot = "Data/initialization/image40_1_114/BSREM_it30/replicate_1/"
# Choose file extension according to extension in subsubroot folder
# file_extension = 'img'
file_extension = 'raw'

import os
# List all files in subsubroot with chosen file extension
filenames = [file for file in os.listdir(subroot + subsubroot) if file.endswith(file_extension)]
# Remove file extension for each file name
filenames = [subroot + subsubroot + filename[:-4] for filename in filenames]

# Pad x and y dimensions. Code need to be extend to pad z dimension
original_shape = (112,112,1)
new_dimx = 114
new_dimy = 114
if (1 in original_shape):
    nb_dimensions = 2
else:
    nb_dimensions = 3
if (nb_dimensions == 2):
    new_shape = (new_dimx,new_dimy,1)
else:
    new_shape = (original_shape[-1],new_dimy,new_dimx)

for filename in filenames:
    path = Path(filename)
    print(path)
    im_full = fijii_np(filename + "." + file_extension,original_shape,type_im='<f')
    im_padded = np.zeros(new_shape,dtype='<f')
    pad_x = (new_dimx - original_shape[0])//2
    pad_y = (new_dimy - original_shape[1])//2
    if (nb_dimensions == 2):
        im_padded[pad_x:-pad_x,pad_y:-pad_y,:] = im_full
    else:
        im_padded[:,pad_y:-pad_y,pad_x:-pad_x] = im_full
    save_img(im_padded,filename + "_114" + "." + file_extension)