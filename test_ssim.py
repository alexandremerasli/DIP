from utils.ssim_python import structural_similarity
from utils.mssim import mssim
import numpy as np
import matplotlib.pyplot as plt
        
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
                        pass
                else: # 3D
                    image = data.reshape(shape[::-1])
            
            fid.close()

    return image


#Loading Ground Truth image to compute metrics
image_gt = fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/database_v2/image4_0/image4_0.raw",shape=((112,112)),type_im='<f')

#Loading corrupted image
image_corrupted = fijii_np("/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/image4_0/BSREM_30it/replicate_1/BSREM_it30.img",shape=((112,112)),type_im='<f')

# plt.imshow(image_gt,cmap='gray')
# plt.show()
# plt.imshow(image_corrupted,cmap='gray')
# plt.show()

# image_gt = 3*np.ones((20,20))
# image_corrupted = 2*np.ones((20,20))

print(mssim(image_gt,image_corrupted,1,1,1))

# Uniform filter
# print(structural_similarity(np.squeeze(image_gt), np.squeeze(image_corrupted), data_range=(image_corrupted).max() - (image_corrupted).min(), use_sample_covariance=False))
# Gaussian filter (Wang et al. 2004)
print(structural_similarity(np.squeeze(image_gt), np.squeeze(image_corrupted), data_range=(image_corrupted).max() - (image_corrupted).min(), sigma=1.5, gaussian_weights=True, use_sample_covariance=False))
# Case in previous nested computation
# print(structural_similarity(np.squeeze(image_gt), np.squeeze(image_corrupted), data_range=(image_corrupted).max() - (image_corrupted).min(), use_sample_covariance=True)) # If use_sample_covariance True, normalize covariances by N-1 rather than N
