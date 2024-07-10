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

subroot = 'data/Algo/'
filenames = [subroot + 'Data/initialization/image40_1/BSREM_it30/replicate_1/BSREM_it30']

original_shape = (112,112)
final_shape = (original_shape[0],original_shape[1],8)

for filename in filenames:
    path = Path(filename)
    print(path)
    img = fijii_np(filename + ".img",original_shape,type_im='<f')
    im_3D = np.zeros(final_shape,dtype='<f')
    for i in range(final_shape[-1]):
        im_3D[:,:,i] = img
    save_img(np.transpose(im_3D.astype(np.float32),axes=(2,1,0)),filename + "_stack.img")