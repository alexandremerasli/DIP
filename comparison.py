## Python libraries

# Pytorch
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--beta', type=str, dest='beta', help='penalty strength (beta)')

# Retrieving arguments in this python script
args = parser.parse_args()
if (args.opti is not None): # Must check if all args are None    
    optimizer = args.opti # CASToR optimizer
    nb_iter = args.nb_iter # number of optimizer iterations
    beta = args.beta # penalty strength (beta)
else: # For VS Code (without command line)
    optimizer = 'BSREM' # CASToR optimizer
    nb_iter = 30 # number of optimizer iterations
    beta = 0.04 # penalty strength (beta)

# Define PET input dimensions according to input data dimensions
PETImage_shape_str = read_input_dim(subroot+'Data/castor_output_it60.hdr')
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

#Loading Ground Truth image to compute metrics
image_gt = fijii_np(subroot+'Data/phantom/phantom_act.img',shape=(PETImage_shape))

## Computing metrics for (must add post smoothing for MLEM) reconstruction

# Metrics arrays

beta = [0.0001,0.001,0.01,0.1,1,10]
beta = [0.01,0.03,0.05,0.07,0.09]
beta = [0.03,0.035,0.04,0.045,0.05]
beta = [0.04]
if (optimizer == 'MLEM'):
    max_iter = 1
elif (optimizer == 'BSREM'):
    max_iter = len(beta)

PSNR_recon = np.zeros(max_iter)
PSNR_norm_recon = np.zeros(max_iter)
MSE_recon = np.zeros(max_iter)
MA_cold_recon = np.zeros(max_iter)
CRC_hot_recon = np.zeros(max_iter)
CRC_bkg_recon = np.zeros(max_iter)
IR_bkg_recon = np.zeros(max_iter)
bias_cold_recon = np.zeros(max_iter)
bias_hot_recon = np.zeros(max_iter)

writer = SummaryWriter()

for i in range(max_iter):

    # castor-recon command line
    header_file = ' -df ' + subroot + 'Data/data_eff10/data_eff10.cdh' # PET data path

    executable = 'castor-recon'
    dim = ' -dim ' + PETImage_shape_str
    vox = ' -vox 4,4,4'
    vb = ' -vb 3'
    it = ' -it ' + str(nb_iter) + ':28'
    th = ' -th 0'
    proj = ' -proj incrementalSiddon'
    psf = ' -conv gaussian,4,4,3.5::psf'

    if (optimizer == 'MLEM'):
        opti = ' -opti ' + optimizer
        conv = ' -conv gaussian,8,8,3.5::post'
        conv = ''
        penalty = ''
        penaltyStrength = ''
    else:
        opti = ' -opti ' + optimizer + ':' + subroot + 'Comparison/BSREM/' + 'BSREM.conf'
        conv = ''
        penalty = ' -pnlt MRF:' + subroot + 'Comparison/BSREM/' + 'MRF.conf'
        penaltyStrength = ' -pnlt-beta ' + str(beta[i])

    output_path = ' -dout ' + subroot + 'Comparison/' + optimizer # Output path for CASTOR framework
    initialimage = ' -img ' + subroot + 'Data/castor_output_it60.hdr'
    initialimage = ''

    # Command line for calculating the Likelihood
    vb_like = ' -vb 0'
    opti_like = ' -opti-fom'

    os.system(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage + penalty + penaltyStrength + conv + psf + ' -fov-out 95')

import subprocess
root = os.getcwd()
test = 24
successful_process = subprocess.call(["python3", root+"/show_castor_results.py", optimizer, str(nb_iter), str(test),str("no suffix here")]) # Showing results in tensorboard