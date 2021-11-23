from torch.utils.tensorboard.writer import SummaryWriter
#from utils_func import input_dim_str_to_list, write_hdr, compute_x_v_u_ADMM, castor_command_line, read_input_dim
import utils_func
from pathlib import Path

import os
import numpy as np
from shutil import copy

def compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path,subdir,i,k,only_x,subroot,subroot_output_path):
    os.system(x_reconstruction_command_line)
    #x = fijii_np(subroot + 'Data/ADMMLim.img',(PETImage_shape[0],PETImage_shape[0]))

    copy(subroot + 'Data/ADMMLim_x.img', full_output_path + '_x.img')
    utils_func.write_hdr([i,k+1],subdir,'x',subroot_output_path=subroot_output_path)

    
    # not changing v
    # true ADMM, changing v
    copy(subroot + 'Data/ADMMLim_v.img', full_output_path + '_v.img')
    if (only_x):
        #copy(subroot_output_path + '/during_eq22/' + format(i) + '_' + format(-1) + '_v.img',full_output_path + '_v.img')
        copy(subroot + 'Data/ADMM_spec_init_v.img', full_output_path + '_v.img')
    
    utils_func.write_hdr([i,k+1],subdir,'x',subroot_output_path=subroot_output_path)
    utils_func.write_hdr([i,k+1],subdir,'v',subroot_output_path=subroot_output_path)
    
    # true ADMM, changing u
    copy(subroot + 'Data/ADMMLim_u.img', full_output_path + '_u.img')
    # not changing u
    if (only_x):
        copy(subroot_output_path + '/during_eq22/' + format(i) + '_' + format(-1) + '_u.img',full_output_path + '_u.img')
    utils_func.write_hdr([i,k+1],subdir,'u',subroot_output_path=subroot_output_path)

# Do not run code if compute_x_v_u_ADMM function is imported in an other file
if __name__ == "__main__":

    root = os.getcwd() # Directory root
    subroot = root + '/data/Algo/'  # Directory subroot

    config = {
    "rho" : 0.0003,
    "alpha" : 0.05,
    "image_init_path_without_extension" : '1_value_cropped',
    "it" : ' -it 1:1',
    "penalty" : ''
    }

    """
    Receiving variables from block 1 part and initializing variables
    """

    suffix = 'ADMMLim' # Suffix to make difference between hyperparameters runs (ADMM from Lim et al. here)
    suffix +=  utils_func.suffix_func(config) # suffix to make difference between raytune runs (different hyperparameters)
    np.save(subroot + 'Config/config' + suffix + '.npy', config) # Save this configuration of hyperparameters, and reload with suffix
    PETImage_shape_str = utils_func.read_input_dim(subroot+'Data/castor_output_it60.hdr')
    PETImage_shape = utils_func.input_dim_str_to_list(PETImage_shape_str)
    config = np.load(subroot + 'Config/config' + suffix + '.npy',allow_pickle='TRUE').item()
    i = 0
    k = -1
    full_output_path = subroot + 'ADMM/' + format(i) + '_' + format(k+1)
    
    subdir = 'ADMM'
    Path(subroot+'ADMM/').mkdir(parents=True, exist_ok=True) # CASTor path
    Path(subroot+'ADMM/' + suffix + '/ADMM').mkdir(parents=True, exist_ok=True) # CASToR path
    
    # To be put in arguments
    image_init_path_without_extension = '1_value_cropped' # Path to initial image for CASToR ADMM reconstruction
    rho = 0 # Penalty strength
    alpha = 0.05 # ADMM parameter
    it = ' -it 60:21'
    penalty = ''

    only_x = False
    subroot_output_path = (subroot + 'ADMM/' + suffix)



    # Compute v^0 save in .hdr format
    x_for_init_v = ' -img ' + subroot + 'Data/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values

    # Useful variables for command line
    u_for_additional_data = ' -additional-data ' + subroot + 'Data/0_sino_value.hdr' # Not used in v^0 computation
    v_for_additional_data = ' -additional-data ' + subroot + 'Data/0_sino_value.hdr' # Not used in v^0 computation






    # Define command line to run ADMM with CASToR
    castor_command_line_x = utils_func.castor_command_line(PETImage_shape_str, alpha, rho, suffix)
    initialimage = ' -img ' + subroot + 'Data/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
    f_mu_for_penalty = ' -multimodal ' + subroot + 'Data/BSREM_it30_REF_cropped.hdr'


    # Compute one ADMM iteration (x, v, u) when only initializing x
    x_reconstruction_command_line = castor_command_line_x + ' -dout ' + subroot_output_path + '/during_eq22' + ' -it 1:1' + x_for_init_v + f_mu_for_penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
    print('vvvvvvvvvvv0000000000')
    compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path,subdir,i,k,only_x,subroot,subroot_output_path)
    utils_func.write_hdr([i,k+1],'ADMM','v',subroot)

    # Compute one ADMM iteration (x, v, u)
    print('xxxxxxxxxxxxxxxxxxxxx')
    u_for_additional_data = ' -additional-data ' + subroot + 'Data/0_sino_value.hdr'
    v_for_additional_data = ' -additional-data ' + full_output_path + '_v.hdr' # Previously computed v^0
    x_reconstruction_command_line = castor_command_line_x + ' -dout ' + subroot_output_path + '/during_eq22' + it + u_for_additional_data + v_for_additional_data + initialimage + f_mu_for_penalty + penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
    compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path,subdir,i,k,only_x,subroot,subroot_output_path)
