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
from utils_func import *

## Arguments for linux command to launch script
# Creating arguments
parser = argparse.ArgumentParser(description='DIP + ADMM computation')
parser.add_argument('--opti', type=str, dest='opti', help='optimizer to use in CASToR')
parser.add_argument('--nb_iter', type=str, dest='nb_iter', help='number of optimizer iterations')
parser.add_argument('--beta', type=str, dest='beta', help='penalty strength (beta)')

# Retrieving arguments in this python script
args = parser.parse_args()
optimizer = args.opti
nb_iter = args.nb_iter
beta = args.beta 

# Define PET input dimensions according to input data dimensions
PETImage_shape_str = read_input_dim()
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)

#Loading Ground Truth image to compute metrics
image_gt = fijii_np(subroot+'Block2/data/phantom_act.img',shape=(PETImage_shape))

## Computing metrics for (must add post smoothing for MLEM) reconstruction

# Metrics arrays

beta = [0.0001,0.001,0.01,0.1,1,10]
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
    it = ' -it ' + str(nb_iter) + ':21'
    th = ' -th 0'
    opti = ' -opti ' + optimizer
    proj = ' -proj incrementalSiddon'

    if (optimizer == 'MLEM'):
        conv = ' -conv gaussian,4,4,3.5::post'
        penalty = ''
        penaltyStrength = ''
    else:
        conv = ''
        penalty = ' -pnlt MRF'
        penaltyStrength = ' -pnlt-beta ' + str(beta[i])

    output_path = ' -dout ' + subroot + 'Comparaison/' + optimizer # Output path for CASTOR framework
    initialimage = ' -img ' + subroot + 'Data/castor_output_it6.hdr'

    # Command line for calculating the Likelihood
    vb_like = ' -vb 0'
    opti_like = ' -opti-fom'

    os.system(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage + penalty + penaltyStrength + conv)

    # load MLEM previously computed image 
    image_optimizer = fijii_np(subroot+'Comparaison/' + optimizer + '/' + optimizer + '_it' + str(nb_iter) + '.img', shape=(PETImage_shape))


    # compute metrics varying beta (if BSREM)
    compute_metrics(image_optimizer,image_gt,i,max_iter,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,writer=writer,write_tensorboard=True)

    # write image varying beta (if BSREM)
    write_image_tensorboard(writer,image_optimizer,"Final image computed with " + optimizer,i) # image in tensorboard

    # Display CRC vs STD curve in tensorboard
    if (i>max_iter - min(max_iter,10)):
        # Creating matplotlib figure
        plt.plot(IR_bkg_recon,CRC_hot_recon,linestyle='None',marker='x')
        plt.xlabel('IR')
        plt.ylabel('CRC')
        # Adding this figure to tensorboard
        writer.flush()
        writer.add_figure('CRC in hot region vs IR in background', plt.gcf(),global_step=i,close=True)
        writer.close()