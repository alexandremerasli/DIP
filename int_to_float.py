import argparse
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
    with open(path + ".s.hdr") as f:
        with open(path + "_float.s.hdr", "w") as f1:
            for line in f:
                if line.strip() == ('number format := signed integer'):
                    f1.write('number format := short float')
                    f1.write('\n')
                elif line.strip() == ('number of bytes per pixel := 2'):
                    f1.write('number of bytes per pixel := 4')
                    f1.write('\n')
                elif line.strip() == ('!name of data file := ' + filename + '.s'):
                    f1.write('!name of data file := ' + filename + '_float.s')
                    f1.write('\n')
                else:
                    f1.write(line)

# Arguments for linux command to launch script
# Creating arguments
parser = argparse.ArgumentParser(description='Converting int binary file to float binary file')
parser.add_argument('--file', type=str, dest='file', help='file to convert')
# Retrieving arguments in this python script
args = parser.parse_args()
path = Path(args.file)
im_int = fijii_np(args.file + ".s",(344,252,1),type='int16')
im_float = np.float32(np.ravel(im_int[1:68516]))
save_img(im_float,args.file + "_float.s")
write_hdr_img(args.file,path.name)