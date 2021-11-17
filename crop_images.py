import numpy as np
from pathlib import Path

def fijii_np(path,shape,type='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
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

filenames = ['data/Algo/Data/1_value','data/Algo/Block2/data/phantom_act','simu_mmr/phantom_act','simu_mmr/phantom_atn','data/Algo/Block2/data/phantom_atn','data/Algo/Data/BSREM_it30_REF','data/Algo/Data/castor_output_it60','data/Algo/Block2/data/random_input']

for filename in filenames:
    path = Path(filename)
    print(path)
    print(filename)
    im_full = fijii_np(filename + ".img",(128,128))
    im_cropped = im_full[8:120,8:120]
    save_img(im_cropped,filename + "_cropped.img")
    if (path.name != 'random_input'):
        print('hdr')
        write_hdr_img(filename,path.name)