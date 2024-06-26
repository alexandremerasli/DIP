import os
import matplotlib.pyplot as plt
import numpy as np
import struct


def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype_np = np.dtype(type_im)
    with open(file_path, 'rb') as fid:
        data = np.fromfile(fid,dtype_np)
        image = data.reshape(shape)
                    
    return image

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

from codecs import decode
import struct

def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 4)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]



import struct

def read_histo_cdf(filename):
    data = []
    with open(filename, 'rb') as f:
        while True:
            # Read 1 uint32 element
            bytes = f.read(4)  # uint32 is 4 bytes
            if not bytes:
                return data
            value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
            data.append(value)

            # Read 5 float32 elements
            for _ in range(5):
                bytes = f.read(4)  # float32 is 4 bytes
                if not bytes:
                    return data
                value = struct.unpack('f', bytes)[0]
                data.append(value)

            # Read 2 uint32 elements
            for _ in range(2):
                bytes = f.read(4)  # uint32 is 4 bytes
                if not bytes:
                    return data
                value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
                data.append(value)

def write_binary_file(data,filename):
    with open(filename, 'wb') as f:
        for i in range(0, len(data), 8):

            # Write 1 uint32 element
            bytes = struct.pack('I', data[i])
            f.write(bytes)
            
            # Write 5 float32 elements
            for j in range(1,5+1):
                bytes = struct.pack('f', data[i+j])
                f.write(bytes)
            
            # Write 2 uint32 elements
            for j in range(6,7+1):
                bytes = struct.pack('I', data[i+j])
                f.write(bytes)



cdf_path = "/home/MEDECINE/mera1140/sherbrooke_workspace/TestCastor/umd_h12_wRot_act_BTB_1_100_df.Cdf"
cdf_path = "data/Algo/Data/database_v2/image40_1/data40_1_1/data40_1_1.cdf"
# cdf_path = "data/Algo/Data/database_v2/image40_1/dataTEST40_1_1/data40_1_1.cdf"
nb_events = int(8308200 / 4) # LP2 data
nb_events = 68516 # simu data
nb_elem_to_read = 8
cdf_np = fijii_np(cdf_path,(nb_events*nb_elem_to_read,1),type_im='<f')

# print(cdf_np[0:100:8])
# print(np.uint32(bin_to_float(cdf_np[0])))
# print("end")

data = read_histo_cdf(cdf_path)

for event in range(nb_events*8):
    # if event%8 == 4:
    #     data[event] *= 100
    if event%8 == 1:
        data[event] = 1
# print(data[4:100:8])
print(data[1:100:8])

write_binary_file(data,"data/Algo/Data/database_v2/image40_1/dataTEST40_1_1/data40_1_1.cdf")


# time 32
# atn
# random
# norm
# event value
# scatter 
# id1 32
# id2 32

