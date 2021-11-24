""""
Libraries
"""

import os
import pytorch_lightning as pl
import numpy as np
from itertools import product
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import time
from shutil import copy

from models.ConvNet3D_real_lightning import ConvNet3D_real_lightning # DIP
from models.ConvNet3D_VAE_lightning import ConvNet3D_VAE_lightning # DIP vae
from models.DD_2D_lightning import DD_2D_lightning # DD
from models.DD_AE_2D_lightning import DD_AE_2D_lightning # DD adding encoder part

import ADMMLim

subroot=os.getcwd()+'/data/Algo/'

def suffix_func(config):
    suffix = "config"
    for key, value in config.items():
        suffix +=  "_" + key + "=" + str(value)
    return suffix

def fijii_np(path,shape,type='<f'):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape)
    return image

'''
def norm_imag(image_corrupt):
    """ Normalization of the 3D input - output [0..1] and the normalization value for each slide"""
    image_corrupt_norm = np.zeros((image_corrupt.shape[0],image_corrupt.shape[1]), dtype=np.float32)
    maxi=np.amax(image_corrupt)
    image_corrupt_norm = image_corrupt / maxi
    return image_corrupt_norm,maxi
'''
def norm_imag(img):
    """ Normalization of the 3D input - output [0..1] and the normalization value for each slide"""
    if (np.max(img) - np.min(img)) != 0:
        return (img - np.min(img)) / (np.max(img) - np.min(img)), np.min(img), np.max(img)
    else:
        return img, np.min(img), np.max(img)

def denorm_imag(img, mini, maxi):
    """ Normalization of the 3D input - output [0..1] and the normalization value for each slide"""
    if (maxi - mini) != 0:
        return img * (maxi - mini) + mini
    else:
        return img

def norm_positive_imag(img):
    """ Normalization of the 3D input - output [0..1] and the normalization value for each slide"""
    if (np.max(img) - np.min(img)) != 0:
        return img / np.max(img), np.min(img), np.max(img)
    else:
        return img, 0, np.max(img)

def denorm_positive_imag(img, mini, maxi):
    """ Normalization of the 3D input - output [0..1] and the normalization value for each slide"""
    if (maxi - mini) != 0:
        return img * maxi 
    else:
        return img

def stand_imag(image_corrupt):
    """ Normalization of the 3D input - output with mean 0 and std 1 for each slide"""
    mean=np.mean(image_corrupt)
    std=np.std(image_corrupt)
    image_center = image_corrupt - mean
    image_corrupt_std = image_center / std

    #return norm_positive_imag(image_corrupt)
    return norm_imag(image_corrupt)
    return image_corrupt_std.detach(),mean,std

def destand_imag(image, mean, std):
    image_np = image.detach().numpy()
    return destand_numpy_imag(image_np, mean, std)

def destand_numpy_imag(image, mean, std):
    #return denorm_positive_imag(image, 0, std)
    return denorm_imag(image, mean, std)
    return image * std + mean

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

def write_hdr(L,subpath,variable_name='',subroot_output_path='',matrix_type='img'):
    """ write a header for the optimization transfer solution (it's use as CASTOR input)"""
    if (len(L) == 1):
        i = L[0]
        if variable_name != '':
            ref_numbers = format(i) + '_' + variable_name
        else:
            ref_numbers = format(i)
    else:
        i = L[0]
        k = L[1]
        if variable_name != '':
            ref_numbers = format(i) + '_' + format(k) + '_' + variable_name
        else:
            ref_numbers = format(i)
    filename = subroot_output_path + '/'+ subpath + '/' + ref_numbers +'.hdr'
    with open(subroot+'Data/castor_output_it60.hdr') as f:
        with open(filename, "w") as f1:
            for line in f:
                if line.strip() == ('!name of data file := castor_output_it60.img'):
                    f1.write('!name of data file := '+ ref_numbers +'.img')
                    f1.write('\n') 
                elif line.strip() == ('patient name := castor_output_it60'):
                    f1.write('patient name := ' + ref_numbers)
                    f1.write('\n') 
                else:
                    if (matrix_type == 'sino'): # There are 68516=2447*28 events, but not really useful for computation
                        if line.strip().startswith('!matrix size [1]'):
                            f1.write('matrix size [1] := 2447')
                            f1.write('\n') 
                        elif line.strip().startswith('!matrix size [2]'):
                            f1.write('matrix size [2] := 28')
                            f1.write('\n')
                        else:
                            f1.write(line) 
                    else:
                        f1.write(line) 

def find_nan(image):
    """ find NaN values on the image"""
    idx = np.argwhere(np.isnan(image))
    print('index with NaN value:',len(idx))
    for i in range(len(idx)):
        image[idx[i,0],idx[i,1]] = 0
    print('index with NaN value:',len(np.argwhere(np.isnan(image))))
    return image

def points_in_circle(center_y,center_x,radius,PETImage_shape,inner_circle=True): # x and y are inverted in an array compared to coordinates
    liste = []   
    center_x = int(PETImage_shape[0]/2 + center_x)
    center_y = int(PETImage_shape[1]/2 + center_y)
    radius = int(radius)
    if (inner_circle): # only looking inside the circle, not on the border
        radius -= 2
    for x, y in product(range(int(radius) + 1 + max(center_x,center_y)), repeat=2):
        if (x-center_x)**2 + (y-center_y)**2 <= radius**2:
            liste.append((x,y))

    return liste

def coord_to_value_array(coord_arr,arr):
    l = np.zeros(len(coord_arr))   
    for i in range(len(coord_arr)):
        coord = coord_arr[i]
        l[i] = arr[coord]
    return l

def compute_metrics(PETImage_shape, image_recon,image_gt,i,PSNR_recon,PSNR_norm_recon,MSE_recon,MA_cold_recon,CRC_hot_recon,CRC_bkg_recon,IR_bkg_recon,bias_cold_recon,bias_hot_recon,writer=None,write_tensorboard=False):

    f_metric = find_nan(image_recon)
    image_gt_norm,mini_gt_input,maxe_gt_input = norm_imag(image_gt)

    # print metrics
    print('Metrics for iteration',i)

    f_metric_norm,mini_f_metrics,maxe_f_metrics = norm_imag(f_metric) # normalizing DIP output
    print('Dif for PSNR calculation',np.amax(f_metric) - np.amin(f_metric),' , must be as small as possible')

    # PSNR calculation
    PSNR_recon[i] = peak_signal_noise_ratio(image_gt, f_metric, data_range=np.amax(f_metric) - np.amin(f_metric)) # PSNR with true values
    PSNR_norm_recon[i] = peak_signal_noise_ratio(image_gt_norm,f_metric_norm) # PSNR with scaled values [0-1]
    print('PSNR calculation', PSNR_norm_recon[i],' , must be as high as possible')

    # MSE calculation
    MSE_recon[i] = np.mean((image_gt - f_metric)**2)
    print('MSE gt', MSE_recon[i],' , must be as small as possible')

    # Contrast Recovery Coefficient calculation    
    # Mean activity in cold cylinder calculation (-c -40. -40. 0. 40. 4. 0.)
    cold_ROI = points_in_circle(-40/4,-40/4,40/4,PETImage_shape)
    cold_ROI_act = coord_to_value_array(cold_ROI,f_metric)
    MA_cold_recon[i] = np.mean(cold_ROI_act)
    #IR_cold_recon[i] = np.std(cold_ROI_act) / MA_cold_recon[i]
    bias_cold_recon[i] = MA_cold_recon[i] - 1.
    print('Mean activity in cold cylinder', MA_cold_recon[i],' , must be close to 0')
    #print('Image roughness in the cold cylinder', IR_cold_recon[i])
    print('FOV bias in cold region',bias_cold_recon[i],' , must be as small as possible')

    # Mean Concentration Recovery coefficient (CRCmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
    hot_ROI = points_in_circle(50/4,10/4,20/4,PETImage_shape)
    hot_ROI_act = coord_to_value_array(hot_ROI,f_metric)
    CRC_hot_recon[i] = np.mean(hot_ROI_act) / 400.
    #IR_hot_recon[i] = np.std(hot_ROI_act) / np.mean(hot_ROI_act)
    bias_hot_recon[i] = CRC_hot_recon[i] - 1.
    print('Mean Concentration Recovery coefficient in hot cylinder', CRC_hot_recon[i],' , must be close to 1')
    #print('Image roughness in the hot cylinder', IR_hot_recon[i])
    print('FOV bias in hot region',bias_hot_recon[i],' , must be as small as possible')

    # Mean Concentration Recovery coefficient (CRCmean) in background calculation (-c 0. 0. 0. 150. 4. 100)
    #bkg_ROI = points_in_circle(0/4,0/4,150/4,PETImage_shape)
    #m0_bkg = (np.sum(coord_to_value_array(bkg_ROI,f_metric)) - np.sum([coord_to_value_array(cold_ROI,f_metric),coord_to_value_array(hot_ROI,f_metric)])) / (len(bkg_ROI) - (len(cold_ROI) + len(hot_ROI)))
    #CRC_bkg_recon[i] = m0_bkg / 100.   
    bkg_ROI = points_in_circle(0/4,100/4,40/4,PETImage_shape)
    bkg_ROI_act = coord_to_value_array(bkg_ROI,f_metric)
    CRC_bkg_recon[i] = np.mean(bkg_ROI_act) / 100.
    IR_bkg_recon[i] = np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)
    print('Mean Concentration Recovery coefficient in background', CRC_bkg_recon[i],' , must be close to 1')
    print('Image roughness in the background', IR_bkg_recon[i],' , must be as small as possible')

    if (write_tensorboard):
        print("Metrics saved in tensorboard")
        writer.add_scalar('MSE gt (best : 0)', MSE_recon[i],i)
        writer.add_scalar('Mean activity in cold cylinder (best : 0)', MA_cold_recon[i],i)
        writer.add_scalar('FOV bias in cold region (best : 0)', bias_cold_recon[i],i)
        writer.add_scalar('Mean Concentration Recovery coefficient in hot cylinder (best : 1)', CRC_hot_recon[i],i)
        writer.add_scalar('FOV bias in hot region (best : 0)', bias_hot_recon[i],i)
        writer.add_scalar('Mean Concentration Recovery coefficient in background (best : 1)', CRC_bkg_recon[i],i)
        writer.add_scalar('Image roughness in the background (best : 0)', IR_bkg_recon[i],i)

def choose_net(net, config):
    if (net == 'DIP'):
        model = ConvNet3D_real_lightning(config) #Loading DIP architecture
        model_class = ConvNet3D_real_lightning #Loading DIP architecture
    elif (net == 'DIP_VAE'):
        model = ConvNet3D_VAE_lightning(config) #Loading DIP VAE architecture
        model_class = ConvNet3D_VAE_lightning #Loading DIP VAE architecture
    else:
        if (net == 'DD'):
            model = DD_2D_lightning(config) #Loading Deep Decoder architecture
            model_class = DD_2D_lightning #Loading Deep Decoder architecture
        elif (net == 'DD_AE'):
            model = DD_AE_2D_lightning(config) #Loading Deep Decoder architecture
            model_class = DD_AE_2D_lightning #Loading Deep Decoder architecture
    return model, model_class

def create_random_input(net,PETImage_shape,config): #CT map for high-count data, but not CT yet...
    if (net == 'DIP' or net == 'DIP_VAE'):
        im_input = np.random.normal(0,1,PETImage_shape[0]*PETImage_shape[1]).astype('float32') # initializing input image with random image (for DIP)
        im_input = im_input.reshape(PETImage_shape) # reshaping (for DIP)
    else:
        if (net == 'DD'):
            input_size_DD = int(PETImage_shape[0] / (2**config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
            im_input = np.random.normal(0,1,config["k_DD"]*input_size_DD*input_size_DD).astype('float32') # initializing input image with random image (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
            im_input = im_input.reshape(config["k_DD"],input_size_DD,input_size_DD) # reshaping (for Deep Decoder) # if original Deep Decoder (i.e. only with decoder part)
            
        elif (net == 'DD_AE'):
            input_size_DD = PETImage_shape[0] # if auto encoder based on Deep Decoder
            im_input = np.random.normal(0,1,input_size_DD*input_size_DD).astype('float32') # initializing input image with random image (for Deep Decoder) # if auto encoder based on Deep Decoder
            im_input = im_input.reshape(input_size_DD,input_size_DD) # reshaping (for Deep Decoder) # if auto encoder based on Deep Decoder

    file_path = (subroot+'Data/initialization/random_input.img')
    save_img(im_input,file_path)

def load_input(net,PETImage_shape,config):
    file_path = (subroot+'Data/initialization/random_input.img') #CT map, but not CT yet...
    if (net == 'DD'):
        input_size_DD = int(PETImage_shape[0] / (2**config["d_DD"])) # if original Deep Decoder (i.e. only with decoder part)
        PETImage_shape = (config['k_DD'],input_size_DD,input_size_DD) # if original Deep Decoder (i.e. only with decoder part)
    elif (net == 'DD_AE'):   
        input_size_DD = PETImage_shape[0] # if auto encoder based on Deep Decoder
        PETImage_shape = (input_size_DD,input_size_DD) # if auto encoder based on Deep Decoder

    im_input = fijii_np(file_path, shape=(PETImage_shape)) # Load input of the DNN (CT image)
    return im_input

def read_input_dim(file_path):
    # Read CASToR header file to retrieve image dimension """
    with open(file_path) as f:
        for line in f:
            if 'matrix size [1]' in line.strip():
                dim1 = [int(s) for s in line.split() if s.isdigit()][-1]
            if 'matrix size [2]' in line.strip():
                dim2 = [int(s) for s in line.split() if s.isdigit()][-1]
            if 'matrix size [3]' in line.strip():
                dim3 = [int(s) for s in line.split() if s.isdigit()][-1]

    # Create variables to store dimensions
    PETImage_shape = (dim1,dim2)
    PETImage_shape_str = str(dim1) + ','+ str(dim2) + ',' + str(dim3)
    if (dim3 > 1):
        raise ValueError("3D not implemented yet")
    print('image shape :', PETImage_shape)
    return PETImage_shape_str

def input_dim_str_to_list(PETImage_shape_str):
    return [int(e.strip()) for e in PETImage_shape_str.split(',')][:-1]

def write_image_tensorboard(writer,image,name,i=0,full_contrast=False):
    # Creating matplotlib figure with colorbar
    if (len(image.shape) != 2):
        print('image is ' + str(len(image.shape)) + 'D, plotting only 2D slice')
        image = image[:,:,0]
    if (full_contrast):
        plt.imshow(image, cmap='gray_r',vmin=np.min(image),vmax=np.max(image)) # Showing each image with maximum contrast  
    else:
        plt.imshow(image, cmap='gray_r',vmin=0,vmax=500) # Showing all images with same contrast
    plt.colorbar()
    #plt.axis('off')
    # Adding this figure to tensorboard
    writer.add_figure(name,plt.gcf(),global_step=i,close=True)# for videos, using slider to change image with global_step

def create_pl_trainer(finetuning, processing_unit, sub_iter_DIP, admm_it, net, checkpoint_simple_path, test, checkpoint_simple_path_exp, name=''):
    from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

    tuning_callback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")
    accelerator = None
    if (processing_unit == 'CPU'): # use cpus and no gpu
        gpus = 0
    elif (processing_unit == 'GPU' or processing_unit == 'both'): # use all available gpus, no cpu (pytorch lightning does not handle cpus and gpus at the same time)
        gpus = -1
        #if (torch.cuda.device_count() > 1):
        #    accelerator = 'dp'

    if (admm_it == 0):
        sub_iter_DIP = 1000 if net.startswith('DD') else 200

    if (finetuning == 'False'): # Do not save and use checkpoints (still save hparams and event files for now ...)
        logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(test), name=name) # Store checkpoints in checkpoint_simple_path path
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_top_k=0, save_weights_only=True) # Do not save any checkpoint (save_top_k = 0)
        trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, callbacks=[checkpoint_callback, tuning_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
    else:
        if (finetuning == 'last'): # last model saved in checkpoint
            # Checkpoints pl variables
            logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(test), name=name) # Store checkpoints in checkpoint_simple_path path
            checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, save_last=True, save_top_k=0) # Only save last checkpoint as last.ckpt (save_last = True), do not save checkpoint at each epoch (save_top_k = 0)
            trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback, tuning_callback],gpus=gpus, accelerator=accelerator,log_gpu_memory="all") # Prepare trainer model with callback to save checkpoint        
        if (finetuning == 'best'): # best model saved in checkpoint
            # Checkpoints pl variables
            logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(test), name=name) # Store checkpoints in checkpoint_simple_path path
            checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_simple_path_exp, filename = 'best_loss', monitor='loss_monitor', save_top_k=1) # Save best checkpoint (save_top_k = 1) (according to minimum loss (monitor)) as best_loss.ckpt
            trainer = pl.Trainer(max_epochs=sub_iter_DIP,log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback, tuning_callback],gpus=gpus, accelerator=accelerator, profiler="simple") # Prepare trainer model with callback to save checkpoint

    return trainer

def load_model(image_net_input_torch, config, finetuning, admm_it, model, model_class, subroot, checkpoint_simple_path_exp, training):
    if (finetuning == 'last'): # last model saved in checkpoint
        if (admm_it > 0): # if model has already been trained
            model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'last.ckpt'), config=config) # Load previous model in checkpoint        
    # if (admm_it == 0):
        # DD finetuning, k=32, d=6
        #model = model_class.load_from_checkpoint(os.path.join(subroot,'high_statistics.ckpt'), config=config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)
        #from torch.utils.tensorboard import SummaryWriter
        #writer = SummaryWriter()
        #out = model(image_net_input_torch)
        #write_image_tensorboard(writer,out.detach().numpy(),"high statistics output)") # Showing all corrupted images with same contrast to compare them together
        #write_image_tensorboard(writer,out.detach().numpy(),"high statistics (" + "output, FULL CONTRAST)",0,full_contrast=True) # Showing each corrupted image with contrast = 1
    
        # Set first network iterations to have convergence, as if we do post processing
        # model = model_class.load_from_checkpoint(os.path.join(subroot,'post_reco'+net+'.ckpt'), config=config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)

    if (finetuning == 'best'): # best model saved in checkpoint
        if (admm_it > 0): # if model has already been trained
            model = model_class.load_from_checkpoint(os.path.join(checkpoint_simple_path_exp,'best_loss.ckpt'), config=config) # Load best model in checkpoint
        #if (admm_it == 0):
        # DD finetuning, k=32, d=6
            #model = model_class.load_from_checkpoint(os.path.join(subroot,'high_statistics.ckpt'), config=config) # Load model coming from high statistics computation (normally coming from finetuning with supervised learning)
        if (training):
            os.system('rm -rf ' + checkpoint_simple_path_exp + '/best_loss.ckpt') # Otherwise, pl will store checkpoint with version in filename
    
    return model

def generate_nn_output(net, config, image_net_input_torch, PETImage_shape, finetuning, admm_it, test, suffix, subroot):
    # Loading using previous model
    model, model_class = choose_net(net, config)
    checkpoint_simple_path_exp = subroot+'Block2/checkpoint/'+format(test) + '/' + suffix_func(config) + '/'
    model = load_model(config, finetuning, admm_it, model, model_class, subroot, checkpoint_simple_path_exp, training=False)

    # Compute output image
    out, mu, logvar, z = model(image_net_input_torch)

    # Loading X_label from block1 to destandardize NN output
    image_corrupt = fijii_np(subroot+'Block2/x_label/' + format(test)+'/'+ format(admm_it - 1) +'_x_label' + suffix + '.img',shape=(PETImage_shape))
    #image_corrupt_norm_scale, mini, maxe = norm_imag(image_corrupt)
    image_corrupt_norm,mean_label,std_label= stand_imag(image_corrupt)

    # Destandardize like at the beginning and add it to list of samples
    out_destand = destand_imag(out, mean_label, std_label)
    return out_destand

def castor_reconstruction(writer, i, castor_command_line_x, subroot, sub_iter_MAP, test, config, suffix, f, mu, PETImage_shape, image_init_path_without_extension):
    only_x = True
    start_time_block1 = time.time()
    mlem_sequence = config['mlem_sequence']
    nb_iter_second_admm = 10

    # Save image f-mu in .img and .hdr format - block 1
    subroot_output_path = (subroot + 'Block1/' + suffix)
    path_before_eq_22 = (subroot_output_path + '/before_eq22/')
    path_during_eq_22 = (subroot_output_path + '/during_eq22/')
    save_img(f-mu, path_before_eq_22 + format(i) + '_f_mu.img')
    write_hdr([i],'before_eq22','f_mu',subroot_output_path)

    # x^0
    copy(subroot + 'Data/initialization/' + image_init_path_without_extension + '.img', path_during_eq_22 + format(i) + '_-1_x.img')
    write_hdr([i,-1],'during_eq22','x',subroot_output_path)

    # Compute u^0 (u^-1 in CASToR) and store it with zeros, and save in .hdr format - block 1            
    u_0 = 0*np.ones((344,252)) # initialize u_0 to zeros
    save_img(u_0,path_during_eq_22 + format(i) + '_-1_u.img')
    write_hdr([i,-1],'during_eq22','u',subroot_output_path,matrix_type='sino')
    
    # Compute v^0 (v^-1 in CASToR) with ADMM_spec_init_v optimizer and save in .hdr format - block 1
    if (i == 0):   # choose initial image for CASToR reconstruction
        x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
        #v^0 is BSREM if we only look at x optimization
        if (only_x):
            x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
        #x_for_init_v = ' -img ' + subroot + 'Data/initialization/' + '1_im_value' + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
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
    ADMMLim.compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,'during_eq22',i,k,only_x,subroot,subroot_output_path)

    '''
    successful_process = subprocess.call(["python3", root+"/ADMMLim.py", str(i), castor_command_line_x, subroot, str(sub_iter_MAP), str(test), suffix, PETImage_shape_str, image_init_path_without_extension, net])
    if successful_process != 0: # if there is an error in block2, then stop the run
        raise ValueError('An error occured in ADMM Lim computation. Stopping overall iterations.')
    x_label = fijii_np(subroot+'Block2/x_label/'+format(test) + '/' + format(i) +'_x_label' + suffix + '.img',shape=(PETImage_shape)) # loading DIP output
    '''  


    # When only initializing x, u computation is only the forward model Ax, thus exactly what we want to initialize v
    copy(subroot + 'Data/ADMMLim_u.img', path_during_eq_22 + format(i) + '_-1_v.img')
    write_hdr([i,-1],'during_eq22','v',subroot_output_path,matrix_type='sino')
        
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
                initialimage = ' -img ' + subroot + 'Data/initialization/' + image_init_path_without_extension + '.hdr' if image_init_path_without_extension != "" else '' # initializing CASToR MAP reconstruction with image_init or with CASToR default values
                #initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i-1) + '_' + format(199) + '_x.hdr'
            elif (i >= 1):
                initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i-1) + '_' + format(nb_iter_second_admm) + '_x.hdr'
                # Trying to initialize ADMMLim
                initialimage = ' -img ' + subroot + 'Data/initialization/' + 'BSREM_it30_REF_cropped.hdr'
                initialimage = ' -img ' + subroot + 'Data/initialization/' + '1_im_value_cropped.hdr'
                if (only_x):
                    initialimage = ' -img ' + subroot + 'Block1/' + suffix + '/during_eq22/' +format(i-1) + '_' + format(nb_iter_second_admm) + '_x.hdr'

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
        ADMMLim.compute_x_v_u_ADMM(x_reconstruction_command_line,full_output_path_k_next,'during_eq22',i,k,only_x,subroot,subroot_output_path=subroot_output_path)

    print("--- %s seconds - second ADMM (CASToR) iteration ---" % (time.time() - start_time_block1))

    # Load previously computed image with CASToR ADMM optimizers
    x = fijii_np(subroot+'Block1/' + suffix + '/during_eq22/' +format(i) + '_' + format (k+1) + '_x.img', shape=(PETImage_shape))

    # Save image x in .img and .hdr format - block 1
    name = (subroot+'Block1/' + suffix + '/out_eq22/' + format(i) + '.img')
    save_img(x, name)
    write_hdr([i],'out_eq22','',subroot_output_path)

    # Save x_label for load into block 2 - CNN as corrupted image (x_label)
    x_label = x + mu
    x_label = find_nan(x_label)

    # Save x_label in .img and .hdr format
    name=(subroot+'Block2/x_label/'+format(test) + '/' + format(i) +'_x_label' + suffix + '.img')
    save_img(x_label, name)

    return x_label

def castor_admm_command_line(PETImage_shape_str, alpha, rho, suffix, only_Lim=False, pnlt=''):
    # castor-recon command line
    header_file = ' -df ' + subroot + 'Data/data_eff10/data_eff10.cdh' # PET data path

    executable = 'castor-recon'
    dim = ' -dim ' + PETImage_shape_str
    vox = ' -vox 4,4,4'
    vb = ' -vb 1'
    th = ' -th 1'
    proj = ' -proj incrementalSiddon'

    opti = ' -opti ADMMLim' + ',' + str(alpha) + ',0.01,10.'
    if (~only_Lim): # DIP + ADMM reconstruction, so choose DIP_ADMM penalty from CASToR
        pnlt = ' -pnlt DIP_ADMM'
    pnlt_beta = ' -pnlt-beta ' + str(rho)

    subroot_output_path_castor = ' -dout ' + subroot + 'Block1/' + suffix + '/' # Output path for CASTOR framework
    input_path = ' -img ' + subroot + 'Block1/' + suffix + '/out_eq22/' # Input path for CASTOR framework

    # Command line for calculating the Likelihood
    opti_like = ' -opti-fom'
    opti_like = ''

    castor_command_line_x = executable + dim + vox + header_file + vb + th + proj + opti_like + pnlt + pnlt_beta + opti # opti must be the last one to be able to change suffix_extended in for loop in ADMM
    return castor_command_line_x

# Do not run code if utils_func.py functions are imported in an other file
if __name__ == "__main__":
    print("This file do not need to be run alone, it just stores useful functions")