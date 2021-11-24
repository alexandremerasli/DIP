# Useful
from pathlib import Path
import os
import numpy as np
from shutil import copy

# Local files to import
import utils_func

def compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path,subdir,i,k,only_x,subroot,subroot_output_path):
    # Compute x,u,v and write hdr file
    os.system(x_reconstruction_command_line)
    #x = fijii_np(subroot + 'Data/ADMMLim.img',(PETImage_shape[0],PETImage_shape[0]))
    copy(subroot + 'Data/ADMMLim_x.img', full_output_path + '_x.img')
    utils_func.write_hdr([i,k+1],subdir,'x',subroot_output_path=subroot_output_path)

    
    # Copy v and write hdr file
    copy(subroot + 'Data/ADMMLim_v.img', full_output_path + '_v.img')
    if (only_x):
        #copy(subroot_output_path + subdir + format(i) + '_' + format(-1) + '_v.img',full_output_path + '_v.img')
        copy(subroot + 'Data/initialization/ADMM_spec_init_v.img', full_output_path + '_v.img')
    utils_func.write_hdr([i,k+1],subdir,'x',subroot_output_path=subroot_output_path)
    utils_func.write_hdr([i,k+1],subdir,'v',subroot_output_path=subroot_output_path)
    
    # Copy u and write hdr file
    copy(subroot + 'Data/ADMMLim_u.img', full_output_path + '_u.img')
    if (only_x):
        copy(subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(-1) + '_u.img',full_output_path + '_u.img')
    utils_func.write_hdr([i,k+1],subdir,'u',subroot_output_path=subroot_output_path)

# Do not run code if compute_x_v_u_ADMM function is imported in an other file
if __name__ == "__main__":

    config = {
    "rho" : 0.0003,
    "alpha" : 0.05, # Put alpha = 1 if True, otherwise too slow. alpha smaller if False
    "image_init_path_without_extension" : '1_im_value_cropped',
    "nb_iter" : 10,
    "nb_subsets" : 21,
    "penalty" : ''
    }

    # Variables from config dictionnary
    image_init_path_without_extension = config["image_init_path_without_extension"] # Path to initial image for CASToR ADMM reconstruction
    rho = config["rho"] # Penalty strength
    alpha = config["alpha"] # ADMM parameter
    it = ' -it ' + str(config["nb_iter"]) + ':' + str(config["nb_subsets"])
    penalty = config["penalty"]
    only_x = False # Freezing u and v computation, just updating x if True

    # Path variables
    root = os.getcwd() # Directory root
    subroot = root + '/data/Algo/'  # Directory subroot
    suffix =  utils_func.suffix_func(config) # Suffix to make difference between raytune runs (different hyperparameters)
    subroot_output_path = (subroot + 'Comparison/ADMMLim/' + suffix)
    i = 0
    k = -2
    full_output_path = subroot_output_path + '/ADMM/' + format(i) + '_' + format(k)
    subdir = 'ADMM'
    Path(subroot+'Comparison/ADMMLim/').mkdir(parents=True, exist_ok=True) # CASTor path
    Path(subroot+'Comparison/ADMMLim/' + suffix + '/ADMM').mkdir(parents=True, exist_ok=True) # CASToR path

    # Save this configuration of hyperparameters, and reload with suffix
    np.save(subroot + 'Config/config' + suffix + '.npy', config)

    # Define PET input dimensions according to input data dimensions
    PETImage_shape_str = utils_func.read_input_dim(subroot+'Data/castor_output_it60.hdr')
    PETImage_shape = utils_func.input_dim_str_to_list(PETImage_shape_str)

    # Define command line to run ADMM with CASToR
    castor_command_line_x = utils_func.castor_admm_command_line(PETImage_shape_str, alpha, rho, suffix)
    initialimage = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
    f_mu_for_penalty = ' -multimodal ' + subroot + 'Data/initialization/BSREM_it30_REF_cropped.hdr'
    x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
    if (only_x):
        x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
        
    # Compute one ADMM iteration (x, v, u) when only initializing x to compute v^0
    if (only_x):
        copy(subroot + 'Data/initialization/0_sino_value.hdr', subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(-1) + '_u.img')
    x_reconstruction_command_line = castor_command_line_x + ' -dout ' + subroot_output_path + '/useless' + ' -it 1:1' + x_for_init_v + f_mu_for_penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
    print('vvvvvvvvvvv0000000000')
    compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path,subdir,i,k-1,only_x,subroot,subroot_output_path)
    utils_func.write_hdr([i,k+1],'ADMM','v',subroot_output_path)

    # Compute one ADMM iteration (x, v, u)
    print('xxxxxxxxxxxxxxxxxxxxx')
    u_for_additional_data = ' -additional-data ' + subroot + 'Data/initialization/0_sino_value.hdr'
    v_for_additional_data = ' -additional-data ' + full_output_path + '_v.hdr' # Previously computed v^0
    x_reconstruction_command_line = castor_command_line_x + ' -dout ' + subroot_output_path + '/' + subdir + it + u_for_additional_data + v_for_additional_data + initialimage + f_mu_for_penalty + penalty # we need f-mu so that ADMM optimizer works, even if we will not use it...
    compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path,subdir,i,k,only_x,subroot,subroot_output_path)
