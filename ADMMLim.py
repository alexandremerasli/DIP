# Useful
from pathlib import Path
import os
import numpy as np
from shutil import copy
import argparse

# Local files to import
import utils_func

def compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path,subdir,i,k,config,only_x,subroot_output_path):
    # Compute x,u,v
    os.system(x_reconstruction_command_line)
    # Write x hdr file
    utils_func.write_hdr([i,k+1],subdir,config,'x',subroot_output_path=subroot_output_path)
    # Write v hdr file and change v file if only x computation is needed
    if (only_x):
        copy(subroot_output_path + subdir + format(i) + '_' + format(-1) + '_v.img',full_output_path + '_v.img')
    utils_func.write_hdr([i,k+1],subdir,config,'v',subroot_output_path=subroot_output_path,matrix_type='sino')
    # Write u hdr file and change u file if only x computation is needed
    if (only_x):
        copy(subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(-1) + '_u.img',full_output_path + '_u.img')
    utils_func.write_hdr([i,k+1],subdir,config,'u',subroot_output_path=subroot_output_path,matrix_type='sino')

# Do not run code if compute_x_v_u_ADMM function is imported in an other file
if __name__ == "__main__":

    config = {
    "rho" : 0.05,
    "alpha" : 0.005, # Put alpha = 1 if True, otherwise too slow. alpha smaller if False
    "image_init_path_without_extension" : '1_im_value_cropped',
    "nb_iter" : 1000, # Number of outer iterations
    "penalty" : 'MRF',
    "image" : 'image0'
    #"penalty" : 'DIP_ADMM'
    }

    # Metrics arrays

    beta = [0.0001,0.001,0.01,0.1,1,10]
    beta = [0.01,0.03,0.05,0.07,0.09]
    beta = [0.03,0.035,0.04,0.045,0.05]
    beta = [0.04]

    optimizer = 'ADMMLim'
    nb_iter = config["nb_iter"]
    max_iter = len(beta)

    PSNR_recon = np.zeros(nb_iter)
    PSNR_norm_recon = np.zeros(nb_iter)
    MSE_recon = np.zeros(nb_iter)
    MA_cold_recon = np.zeros(nb_iter)
    CRC_hot_recon = np.zeros(nb_iter)
    CRC_bkg_recon = np.zeros(nb_iter)
    IR_bkg_recon = np.zeros(nb_iter)

    ## Arguments for linux command to launch script
    # Creating arguments
    parser = argparse.ArgumentParser(description='ADMM from Lim et al. computation')
    parser.add_argument('--nb_iter_x', type=int, dest='nb_iter_x', help='number of x iterations')
 
    # Retrieving arguments in this python script
    args = parser.parse_args()

    # For VS Code (without command line)
    if (args.nb_iter_x is None): # Must check if all args are None
        args.nb_iter_x = '10' # Lim et al. does only 1 iteration
        args.nb_subsets = '1' # Lim et al. does not use subsets, so put number of subsets to 1

    # Path variables
    root = os.getcwd() # Directory root
    subroot = root + '/data/Algo/'  # Directory subroot
    suffix =  utils_func.suffix_func(config) # Suffix to make difference between raytune runs (different hyperparameters)
    subroot_output_path = (subroot + 'Comparison/ADMMLim/' + suffix)
    i = 0
    k = -2
    full_output_path_k_next = subroot_output_path + '/ADMM/' + format(i) + '_' + format(k+1)
    subdir = 'ADMM'
    Path(subroot+'Comparison/ADMMLim/').mkdir(parents=True, exist_ok=True) # CASTor path
    Path(subroot+'Comparison/ADMMLim/' + suffix + '/ADMM').mkdir(parents=True, exist_ok=True) # CASToR path

    # Variables from config dictionnary
    image_init_path_without_extension = config["image_init_path_without_extension"] # Path to initial image for CASToR ADMM reconstruction
    rho = config["rho"] # Penalty strength
    alpha = config["alpha"] # ADMM parameter
    it = ' -it ' + str(args.nb_iter_x) + ':' + str(args.nb_subsets)
    penalty = ' -pnlt ' + config["penalty"]
    if config["penalty"] == "MRF":
        penalty += ':' + subroot + 'Comparison/ADMMLim/' + 'MRF.conf'
    only_x = False # Freezing u and v computation, just updating x if True

    # Save this configuration of hyperparameters, and reload with suffix
    np.save(subroot + 'Config/config' + suffix + '.npy', config)

    # Define PET input dimensions according to input data dimensions
    PETImage_shape_str = utils_func.read_input_dim(subroot+'Data/database_v2/' + config["image"] + '/' + config["image"] + '.hdr')
    PETImage_shape = utils_func.input_dim_str_to_list(PETImage_shape_str)

    #Loading Ground Truth image to compute metrics
    image_gt = utils_func.fijii_np(subroot+'Data/database_v2/' + config["image"] + '/' + config["image"] + '.raw',shape=(PETImage_shape))

    # Initialize u^0 (u^-1 in CASToR)
    copy(subroot + 'Data/initialization/0_sino_value.hdr', full_output_path_k_next + '_u.hdr')
    utils_func.write_hdr([i,-1],subdir,config,'u',subroot_output_path,matrix_type='sino')

    # Define command line to run ADMM with CASToR, to compute v^0
    castor_command_line_x = utils_func.castor_admm_command_line(PETImage_shape_str, alpha, rho, config ,True, penalty)
    initialimage = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
    f_mu_for_penalty = ' -multimodal ' + subroot + 'Data/initialization/BSREM_it30_REF_cropped.hdr'
    x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
    if (only_x):
        x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
        
    # Compute one ADMM iteration (x, v, u) when only initializing x to compute v^0
    if (only_x):
        copy(subroot + 'Data/initialization/0_sino_value.hdr', subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(-1) + '_u.img')
    x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + ' -it 1:1' + x_for_init_v + f_mu_for_penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
    print('vvvvvvvvvvv0000000000')
    compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,subdir,i,k-1,config,only_x,subroot_output_path)
    utils_func.write_hdr([i,k+1],'ADMM',config,'v',subroot_output_path,matrix_type='sino')

    # Compute one ADMM iteration (x, v, u)
    print('xxxxxxxxxxxxxxxxxxxxx')
    for k in range(-1,config["nb_iter"]):
        # Initialize variables for command line
        if (k == -1):
            if (i == 0):   # choose initial image for CASToR reconstruction
                initialimage = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
        else:
            initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(k) + '_x.hdr'

        base_name_k = format(i) + '_' + format(k)
        base_name_k_next = format(i) + '_' + format(k+1)
        full_output_path_k = subroot_output_path + '/' + subdir + '/' + base_name_k
        full_output_path_k_next = subroot_output_path + '/' + subdir + '/' + base_name_k_next
        v_for_additional_data = ' -additional-data ' + full_output_path_k + '_v.hdr'
        u_for_additional_data = ' -additional-data ' + full_output_path_k + '_u.hdr'

        # Compute one ADMM iteration (x, v, u)
        x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + it + u_for_additional_data + v_for_additional_data + initialimage + f_mu_for_penalty + penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
        compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,subdir,i,k,config,only_x,subroot_output_path)

    import subprocess
    root = os.getcwd()
    test = 24
    successful_process = subprocess.call(["python3", root+"/show_castor_results.py", optimizer, str(nb_iter), str(test),suffix]) # Showing results in tensorboard