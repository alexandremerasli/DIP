import numpy as np
from pathlib import Path

def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""               
    file_path=(path)
    nb_dimensions = len(shape)
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

subroot = 'data/Algo/'
filenames = [subroot + 'Data/initialization/image010_3D/BSREM_30it/replicate_1/BSREM_it30']

original_shape = (192,192,184)
new_dimx = 284
new_dimy = 284
if (len(original_shape) == 2):
    new_shape = (new_dimx,new_dimy)
else:
    new_shape = (original_shape[-1],new_dimy,new_dimx)

for filename in filenames:
    path = Path(filename)
    print(path)
    im_full = fijii_np(filename + ".img",original_shape,type_im='<f')
    im_padded = np.zeros(new_shape,dtype='<f')
    pad_x = (new_dimx - original_shape[0])//2
    pad_y = (new_dimy - original_shape[1])//2
    im_padded[:,pad_y:-pad_y,pad_x:-pad_x] = im_full
    save_img(im_padded,filename + "_padded.img")