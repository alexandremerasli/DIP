import numpy as np
import matplotlib.pyplot as plt

import argparse

def fijii_np(path,shape,type_im='<f'):
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    if (1 in shape): # 2D
        image = data.reshape(shape)
    else: # 3D
        image = data.reshape(shape[::-1])

    return image

parser = argparse.ArgumentParser(description='Display absolute difference between 2 images')
parser.add_argument('--img1', type=str, dest='img1', help='first image .img')
parser.add_argument('--img2', type=str, dest='img2', help='second image .img')

args = parser.parse_args()

root = 'data/Algo/'
PETImage_shape = (112,112,1)
PETImage_shape = (70,70,55)
PETImage_shape = (200,200,109)
PETImage_shape = (172,172,127)


args.img1 = "data/Algo/image010_3D/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.03_adapt=nothing_mu_DI=50_tau_D=50_lr=0.01_sub_i=300_overr=True_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/out_DIP-1_FINAL.img"
args.img2 = "data/Algo/image010_3D/replicate_1/nested/Block2/config_image=BSREM_it30_rho=0.03_adapt=nothing_mu_DI=50_tau_D=50_lr=0.01_sub_i=300_overr=True_opti_=Adam_skip_=3_scali=standardization_input=CT_nb_ou=10_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=0_mlem_=False/out_cnn/24/out_DIP98_FINAL.img"





img1_np = fijii_np(args.img1, shape=(PETImage_shape))
print("min 1 = " ,np.min(img1_np))
print("mean 1 = " ,np.mean(img1_np))
print("max 1 = " ,np.max(img1_np))
img2_np = fijii_np(args.img2, shape=(PETImage_shape))
print("")
print("min 2 = " ,np.min(img2_np))
print("mean 2 = " ,np.mean(img2_np))
print("max 2 = " ,np.max(img2_np))

img1_np = img1_np[55,:,:]
img2_np = img2_np[55,:,:]

#'''
plt.figure()
#plt.imshow(np.abs(img1_np - img2_np), cmap='gray_r')
plt.imshow(img1_np - img2_np, cmap='bwr')
plt.title('absolute difference between img1 and img2 (1-2)')
plt.colorbar()
plt.savefig(root+'diff_img.png')

plt.figure()
plt.imshow(img1_np / img2_np, cmap='bwr')

#'''
#'''
for i in range(img1_np.shape[0]):
    for j in range(img1_np.shape[1]):
        if (np.abs(img1_np[i,j] / img2_np[i,j] > 100)):
            print("img1_np = ",img1_np[i,j])
            print("img2_np = ",img2_np[i,j])
#'''
#'''
plt.title('relative difference between img1 and img2')
plt.colorbar()
plt.savefig(root+'relative_diff_img.png')
#'''

MSE_normed = np.linalg.norm(img1_np - img2_np) / (PETImage_shape[0]*PETImage_shape[1]*PETImage_shape[2])
print("MSE : ",MSE_normed)
print("Numerical error below threshold : ",MSE_normed < 1e-5)

plt.figure()
#plt.imshow(np.abs(img1_np), cmap='gray_r')
#plt.imshow(img1_np, cmap='gray_r',vmin=np.min(img1_np),vmax=np.max(img1_np))
print(img1_np.shape)
plt.imshow(img1_np, cmap='gray_r',vmin=0,vmax=12000)
plt.title('img1')
plt.colorbar()
plt.savefig(root+'img1.png')

plt.figure()
#plt.imshow(np.abs(img2_np), cmap='gray_r')
#plt.imshow(img2_np, cmap='gray_r',vmin=np.min(img1_np),vmax=np.max(img1_np))
plt.imshow(img2_np, cmap='gray_r',vmin=0,vmax=12000)
plt.title('img2')
plt.colorbar()
plt.savefig(root+'img2.png')