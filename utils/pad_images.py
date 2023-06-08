import numpy as np
from pathlib import Path

def fijii_np(path,shape,type_im=None):
    """"Transforming raw data to numpy array"""


    attempts = 0

    while attempts < 1000:
        attempts += 1
        try:
            type_im = ('<f')*(type_im=='<f') + ('<d')*(type_im=='<d')
            file_path=(path)
            dtype_np = np.dtype(type_im)
            with open(file_path, 'rb') as fid:
                data = np.fromfile(fid,dtype_np)
                if (1 in shape): # 2D
                    #shape = (shape[0],shape[1])
                    image = data.reshape(shape)
                else: # 3D
                    image = data.reshape(shape[::-1])
            attempts = 1000
            break
        except:
            # fid.close()
            type_im = ('<f')*(type_im=='<d') + ('<d')*(type_im=='<f')
            file_path=(path)
            dtype_np = np.dtype(type_im)
            with open(file_path, 'rb') as fid:
                data = np.fromfile(fid,dtype_np)
                if (1 in shape): # 2D
                    #shape = (shape[0],shape[1])
                    try:
                        image = data.reshape(shape)
                    except Exception as e:
                        # print(data.shape)
                        # print(type_im)
                        # print(dtype_np)
                        # print(fid)
                        # '''
                        # import numpy as np
                        # data = fromfile(fid,dtype('<f'))
                        # np.save('data' + str(self.replicate) + '_' + str(attempts) + '_f.npy', data)
                        # '''
                        # print('Failed: '+ str(e) + '_' + str(attempts))
                        pass
                else: # 3D
                    image = data.reshape(shape[::-1])
            
            fid.close()
        '''
        image = data.reshape(shape)
        #image = transpose(image,axes=(1,2,0)) # imshow ok
        #image = transpose(image,axes=(1,0,2)) # imshow ok
        #image = transpose(image,axes=(0,1,2)) # imshow ok
        #image = transpose(image,axes=(0,2,1)) # imshow ok
        #image = transpose(image,axes=(2,0,1)) # imshow ok
        #image = transpose(image,axes=(2,1,0)) # imshow ok
        '''
        
    #'''
    #image = data.reshape(shape)
    '''
    try:
        print(image[0,0])
    except Exception as e:
        print('exception image: '+ str(e))
    '''
    # print("read from ", path)
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

filenames = ['data/Algo/Data/initialization/image010_3D/BSREM_30it/replicate_1/BSREM_it30']

for filename in filenames:
    path = Path(filename)
    print(path)
    im_full = fijii_np(filename + ".img",(230,150,127),type_im='<f')
    im_padded = np.zeros((127,152,232),dtype='<f')
    im_padded[:,1:-1,1:-1] = im_full
    save_img(im_padded,filename + "_padded.img")