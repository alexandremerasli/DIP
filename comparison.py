## Python libraries

# Useful
import os
import argparse

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils_func import read_input_dim, input_dim_str_to_list, fijii_np, subroot

## Arguments for linux command to launch script
# Creating arguments
parser = argparse.ArgumentParser(description='DIP + ADMM computation')
parser.add_argument('--opti', type=str, dest='opti', help='optimizer to use in CASToR')
parser.add_argument('--nb_iter', type=str, dest='nb_iter', help='number of optimizer iterations')
parser.add_argument('--beta', type=str, dest='beta', help='penalty strength (beta)',nargs='+')
parser.add_argument('--image', type=str, dest='image', help='phantom image from database')

# Retrieving arguments in this python script
args = parser.parse_args()
if (args.opti is not None): # Must check if all args are None    
    optimizer = args.opti # CASToR optimizer
    nb_iter = args.nb_iter # number of optimizer iterations
    beta = args.beta # penalty strength (beta)
    image = args.image # phantom image from database
else: # For VS Code (without command line)
    optimizer = 'BSREM' # CASToR optimizer
    nb_iter = 10 # number of optimizer iterations
    #beta = 0.04 # penalty strength (beta)
    beta = list(np.logspace(-4,1,num=6)) # Good try for image0
    beta = list(np.logspace(-7,-5,num=6)) # Good try for image1
    #beta = [0.01,0.03,0.05,0.07,0.09]
    #beta = [0.03,0.035,0.04,0.045,0.05]
    #beta = [0.04]
    image = "image1"

# Define PET input dimensions according to input data dimensions
PETImage_shape_str = read_input_dim(subroot+'Data/database_v2/' + image + '/' + image + '.hdr')
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

#Loading Ground Truth image to compute metrics
image_gt = fijii_np(subroot+'Data/database_v2/' + image + '/' + image + '.raw',shape=(PETImage_shape))

## Computing metrics for (must add post smoothing for MLEM) reconstruction

# Metrics arrays

if (optimizer == 'MLEM'):
    beta = [0]
max_iter = len(beta)

PSNR_recon = np.zeros(max_iter)
PSNR_norm_recon = np.zeros(max_iter)
MSE_recon = np.zeros(max_iter)
MA_cold_recon = np.zeros(max_iter)
CRC_hot_recon = np.zeros(max_iter)
CRC_bkg_recon = np.zeros(max_iter)
IR_bkg_recon = np.zeros(max_iter)

for i in range(max_iter):
    print(i)

    # castor-recon command line
    header_file = ' -df ' + subroot + 'Data/database_v2/' + image + '/data' + image[-1] + '/data' + image[-1]  + '.cdh' # PET data path

    executable = 'castor-recon'
    dim = ' -dim ' + PETImage_shape_str
    vox = ' -vox 4,4,4'
    vb = ' -vb 1'
    it = ' -it ' + str(nb_iter) + ':28'
    th = ' -th 0'
    proj = ' -proj incrementalSiddon'
    psf = ' -conv gaussian,4,4,3.5::psf'

    if (optimizer == 'MLEM'):
        opti = ' -opti ' + optimizer
        conv = ' -conv gaussian,8,8,3.5::post'
        #conv = ''
        penalty = ''
        penaltyStrength = ''
    else:
        opti = ' -opti ' + optimizer + ':' + subroot + 'Comparison/BSREM/' + 'BSREM.conf'
        conv = ''
        penalty = ' -pnlt MRF:' + subroot + 'Comparison/BSREM/' + 'MRF.conf'
        penaltyStrength = ' -pnlt-beta ' + str(beta[i])

    output_path = ' -dout ' + subroot + 'Comparison/' + optimizer + '_beta_' + str(beta[i]) # Output path for CASTOR framework
    initialimage = ' -img ' + subroot+'Data/database_v2/' + image + '/' + image + '.hdr'
    initialimage = ''

    # Command line for calculating the Likelihood
    vb_like = ' -vb 0'
    opti_like = ' -opti-fom'

    print("CASToR command line :")
    print("")
    print(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage + penalty + penaltyStrength + conv + psf)
    print("")
    os.system(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage + penalty + penaltyStrength + conv + psf) # + ' -fov-out 95')

import subprocess
root = os.getcwd()
test = 24
successful_process = subprocess.call(["python3", root+"/show_castor_results.py", optimizer, str(nb_iter), str(test),str("no suffix here"),image,str(beta)]) # Showing results in tensorboard