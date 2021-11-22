from torch.utils.tensorboard.writer import SummaryWriter
from utils_func import input_dim_str_to_list, save_img, write_hdr, fijii_np, write_image_tensorboard, find_nan, compute_x_v_u_ADMM

import sys
import numpy as np
import time
from shutil import copy

def castor_reconstruction(writer, i, castor_command_line_x, castor_command_line_init_v, subroot, sub_iter_MAP, test, subroot_output_path_castor, config, suffix, f, mu, PETImage_shape, image_init_path_without_extension):
    only_x = False
    start_time_block1 = time.time()
    mlem_sequence = config['mlem_sequence']
    nb_iter_second_admm = 10

    # Save image f-mu in .img and .hdr format - block 1
    subroot_output_path = (subroot + 'Block1/' + suffix)
    path_before_eq_22 = (subroot_output_path + '/before_eq22/')
    path_during_eq_22 = (subroot_output_path + '/during_eq22/')
    save_img(f-mu, path_before_eq_22 + format(i) + '_f_mu.img')
    write_hdr([i],config,'before_eq22','f_mu',subroot)

    # x^0
    copy(subroot + 'Data/' + image_init_path_without_extension + '.img', path_during_eq_22 + format(i) + '_-1_x.img')
    write_hdr([i,-1],config,'during_eq22','x',subroot)

    # Compute u^0 (u^-1 in CASToR) and store it with zeros, and save in .hdr format - block 1            
    u_0 = 0*np.ones((344,252)) # initialize u_0 to zeros
    save_img(u_0,path_during_eq_22 + format(i) + '_-1_u.img')
    write_hdr([i,-1],config,'during_eq22','u',subroot)
    
    # Compute v^0 (v^-1 in CASToR) with ADMM_spec_init_v optimizer and save in .hdr format - block 1
    if (i == 0):   # choose initial image for CASToR reconstruction
        x_for_init_v = ' -img ' + subroot + 'Data/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
        #v^0 is BSREM if we only look at x optimization
        if (only_x):
            x_for_init_v = ' -img ' + subroot + 'Data/' + 'BSREM_it30_REF' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
        #x_for_init_v = ' -img ' + subroot + 'Data/' + '1_value' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
    elif (i >= 1):
        x_for_init_v = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i-1) + '_' + format(nb_iter_second_admm) + '_x.hdr'
    
    # Useful variables for command line
    k=-3
    full_output_path_k = subroot_output_path + '/during_eq22/' + format(i) + '_' + format(k)
    full_output_path_k_next = subroot_output_path + '/during_eq22/' + format(i) + '_' + format(k+1)
    f_mu_for_penalty = ' -multimodal ' + subroot_output_path + '/before_eq22/' + format(i) + '_f_mu' + '.hdr'
    v_for_additional_data = ' -additional-data ' + full_output_path_k + '_v.hdr'
    u_for_additional_data = ' -additional-data ' + full_output_path_k + '_u.hdr'

    # Compute one ADMM iteration (x, v, u) when only initializing x
    x_reconstruction_command_line = castor_command_line_x + ' -dout ' + subroot_output_path + '/during_eq22' + ' -it 1:1' + x_for_init_v + f_mu_for_penalty #+ u_for_additional_data + v_for_additional_data # we need f-mu so that ADMM optimizer works, even if we will not use it...
    print('vvvvvvvvvvv0000000000')
    compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,config,i,k,suffix,only_x,subroot)
    # When only initializing x, u computation is only the forward model Ax, thus exactly what we want to initialize v
    copy(path_during_eq_22 + format(i) + '_-2_u.img', path_during_eq_22 + format(i) + '_-1_v.img')
    write_hdr([i,-1],config,'during_eq22','v',subroot)
        
    # Choose number of argmax iteration for (second) x computation
    if (mlem_sequence):
        it = ' -it 2:56,4:42,6:36,4:28,4:21,2:14,2:7,2:4,2:2,2:1' # large subsets sequence to approximate argmax
    else:
        it = ' -it ' + str(sub_iter_MAP) + ':1' # Only 2 iterations to compute argmax, if we estimate it is an enough precise approximation 
        it = ' -it ' + '5:14' # Only 2 iterations to compute argmax, if we estimate it is an enough precise approximation 

    # Second ADMM computation
    for k in range(-1,nb_iter_second_admm):
        # Initialize variables for command line
        if (k == -1):
            if (i == 0):   # choose initial image for CASToR reconstruction
                initialimage = ' -img ' + subroot + 'Data/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
                #initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i-1) + '_' + format(199) + '_x.hdr'
            elif (i >= 1):
                initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i-1) + '_' + format(nb_iter_second_admm) + '_x.hdr'
                # Trying to initialize ADMMLim
                initialimage = ' -img ' + subroot + 'Data/' + 'BSREM_it30_REF_cropped.hdr'

        else:
            initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i) + '_' + format(k) + '_x.hdr'

        full_output_path_k = subroot_output_path + '/during_eq22/' + format(i) + '_' + format(k)
        full_output_path_k_next = subroot_output_path + '/during_eq22/' + format(i) + '_' + format(k+1)
        f_mu_for_penalty = ' -multimodal ' + subroot_output_path + '/before_eq22/' + format(i) + '_f_mu' + '.hdr'
        v_for_additional_data = ' -additional-data ' + full_output_path_k + '_v.hdr'
        u_for_additional_data = ' -additional-data ' + full_output_path_k + '_u.hdr'

        x = fijii_np(full_output_path_k + '_x.img', shape=(PETImage_shape[0],PETImage_shape[1]))
        if (k>=0):
            write_image_tensorboard(writer,x,"x in second ADMM over iterations", k+i*nb_iter_second_admm) # Showing all corrupted images with same contrast to compare them together
            write_image_tensorboard(writer,x,"x in second ADMM over iterations(FULL CONTRAST)", k+i*nb_iter_second_admm,full_contrast=True) # Showing all corrupted images with same contrast to compare them together

        # Compute one ADMM iteration (x, v, u)
        x_reconstruction_command_line = castor_command_line_x + ' -dout ' + subroot_output_path + '/during_eq22' + it + f_mu_for_penalty + u_for_additional_data + v_for_additional_data + initialimage
        compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,config,i,k,suffix,only_x,subroot=subroot)

    print("--- %s seconds - second ADMM (CASToR) iteration ---" % (time.time() - start_time_block1))

    # Load previously computed image with CASToR ADMM optimizers
    x = fijii_np(subroot+'Block1/' + suffix + '/during_eq22/' +format(i) + '_' + format (k+1) + '_x.img', shape=(PETImage_shape))

    # Save image x in .img and .hdr format - block 1
    name = (subroot+'Block1/' + suffix + '/out_eq22/' + format(i) + '.img')
    save_img(x, name)
    write_hdr([i], config, 'out_eq22','',subroot)

    # Save x_label for load into block 2 - CNN as corrupted image (x_label)
    x_label = x + mu
    x_label = find_nan(x_label)

    # Save x_label in .img and .hdr format
    name=(subroot+'Block2/x_label/'+format(test) + '/' + format(i) +'_x_label' + suffix + '.img')
    save_img(x_label, name)

    return x_label

"""
Receiving variables from block 1 part and initializing variables
"""

i = int(sys.argv[1]) # Current ADMM iteration 
castor_command_line_x = sys.argv[2] # Label of the experiment
castor_command_line_init_v = sys.argv[3] # Network architecture
subroot = sys.argv[4] # Processing unit (CPU or GPU)
sub_iter_MAP = int(sys.argv[5]) # Finetuning (with best model or last saved model for checkpoint) or not for the DIP optimizations
test = int(sys.argv[6]) # PET input dimensions (string, tF)
subroot_output_path_castor = sys.argv[7] # Directory root
suffix =  sys.argv[8] # Suffix to make difference between hyperparameters runs
PETImage_shape_str =sys.argv[9]
image_init_path_without_extension = sys.argv[10]
net = sys.argv[11]

writer = SummaryWriter()
PETImage_shape = input_dim_str_to_list(PETImage_shape_str)
config = np.load(subroot + 'Config/config' + suffix + '.npy',allow_pickle='TRUE').item()
f = fijii_np(subroot+'Block2/out_cnn/'+ format(test)+'/out_' + net + '' + format(i-1) + suffix + '.img',shape=(PETImage_shape)) # loading DIP output
mu = fijii_np(subroot+'Block2/mu/'+ format(test)+'/mu_' + format(i-1) + suffix + '.img',shape=(PETImage_shape)) # loading mu

x_label = castor_reconstruction(writer, i, castor_command_line_x, castor_command_line_init_v, subroot, sub_iter_MAP, test, subroot_output_path_castor, config, suffix, f, mu, PETImage_shape, image_init_path_without_extension)