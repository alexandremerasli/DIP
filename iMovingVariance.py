from torch import from_numpy
from numpy import inf, zeros, float64, squeeze, newaxis, ones_like, amax, amin, mean, ones, NaN, transpose
from numpy.linalg import norm

# Local files to import
from vGeneral import vGeneral

class iMovingVariance(vGeneral):
    def __init__(self, config):
        # super().__init__(config)
        print("init")

    def initializeSpecific(self,config,root, *args, **kwargs):

        ## Variables for MV ##
        self.queueQ = []
        self.VAR_min = inf
        self.SUCCESS = False
        self.stagnate = 0

        self.patienceNumber = config["patienceNumber"]
        self.epochStar = -1
        self.VAR_recon = []
        self.MSE_MV = []
        self.PSNR_MV = []
        self.SSIM_MV = []
        self.DIP_it_if_no_ES_found = config["DIP_it_if_no_ES_found"]
        

        self.EMV_or_WMV = config["EMV_or_WMV"]
        if self.EMV_or_WMV == "EMV":    
            self.EMA = zeros((self.PETImage_shape))
            self.EMV = 0
            self.alpha_EMV = config["alpha_EMV"]
        else:
            # self.MV = 0
            self.windowSize = config["windowSize"]

        #self.queueQ = array((self.windowSize,self.PETImage_shape))

        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(float64)

        # Load phantom ROI
        if (self.simulation): # 2D
            self.phantom_ROI = self.get_phantom_ROI(self.phantom)
        else: # 3D
            if (self.phantom == "image010_3D"): # ROI was defined by thresholding PET BSREM
                self.phantom_ROI = self.get_phantom_ROI(self.phantom)
            else:
                self.phantom_ROI = ones(self.PETImage_shape)

    def runComputation(self,config,root):
        pass

    def compute_MV_value(self,out,epoch,sub_iter_DIP,queueQ,SUCCESS,VAR_min,stagnate,descale=True,MV_value_csv=NaN,current_DIP_iteration=0, MV_metrics_already_stored_in_csv=False):
        
        if (not MV_metrics_already_stored_in_csv):
            # Descale, squeeze image and add 3D dimension to 1 (ok for 2D images)
            if (descale):
                out = self.descale_imag(from_numpy(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            out = squeeze(out)
            if (self.nb_dimensions == 2): # 2D, add new axis because squeeze before (needed because several dimensions could be present because of torch tensors)
                out = out[:,:,newaxis]
            else: # 3D
                out = transpose(out,axes=(1,2,0))
                image_gt_reversed = transpose(self.image_gt,axes=(1,2,0))
                phantom_ROI_reversed = transpose(self.phantom_ROI,axes=(1,2,0))
            
            # Crop image to inside phantom if simulations
            if (self.nb_dimensions == 2):
                out_cropped = out * self.phantom_ROI
                image_gt_cropped = self.image_gt * self.phantom_ROI
            else:

                # Crop image to inside phantom
                # out = out * self.phantom_ROI.reshape(self.phantom_ROI.shape[::-1])
                out = out * phantom_ROI_reversed
                # image_gt_reversed = image_gt_reversed * self.phantom_ROI.reshape(self.phantom_ROI.shape[::-1])
                image_gt_reversed = image_gt_reversed * phantom_ROI_reversed

                # Taking only slice from 3D data
                # out = out[:,:,int(out.shape[2]/2)+20]
                # image_gt_reversed = image_gt_reversed[:,:,int(image_gt_reversed.shape[2]/2)+20]
                slice_MV = out.shape[2]//2 # hard coded
                out = out[:,:,slice_MV]
                image_gt_reversed = image_gt_reversed[:,:,slice_MV]
                if (len(self.EMA.shape) == 3):
                    # self.EMA = self.EMA[:,:,int(self.EMA.shape[2]/2)+20]
                    self.EMA = self.EMA[:,:,slice_MV]


                out_cropped = out
                image_gt_cropped = image_gt_reversed

            from skimage.metrics import peak_signal_noise_ratio
            from skimage.metrics import structural_similarity

            append_metrics = False
            if (self.EMV_or_WMV == "WMV"):
                if (len(queueQ) == self.windowSize - 1):
                    append_metrics = True
            else:
                append_metrics = True

            if append_metrics:
                self.MSE_MV.append(mean((image_gt_cropped - out_cropped)**2))
                self.PSNR_MV.append(peak_signal_noise_ratio(image_gt_cropped, out_cropped, data_range=amax(out_cropped) - amin(out_cropped)))
                self.SSIM_MV.append(structural_similarity(squeeze(image_gt_cropped), squeeze(out_cropped), data_range=out_cropped.max() - out_cropped.min()))

        if (self.EMV_or_WMV == "WMV"):
            #####################################  Window Moving Variance  #############################################
            if (not MV_metrics_already_stored_in_csv):
                queueQ.append(out_cropped.flatten()) # Add last computed image to last element in queueQ from window
                if (len(queueQ) == self.windowSize):
                    # Compute mean for this window
                    mean_im = queueQ[0].copy()
                    for x in queueQ[1:self.windowSize]:
                        mean_im += x
                    mean_im = mean_im / self.windowSize
                    # Compute variance for this window
                    VAR = norm(queueQ[0] - mean_im) ** 2
                    for x in queueQ[1:self.windowSize]:
                        VAR += norm(x - mean_im) ** 2
                    VAR = VAR / self.windowSize
                    # Check if current variance is smaller than minimum previously computed variance, else count number of iterations since this minimum
                    if VAR < VAR_min and not SUCCESS:
                        VAR_min = VAR
                        self.epochStar = epoch  # current detection point
                        stagnate = 1
                    else:
                        stagnate += 1
                    # ES point has been found
                    if stagnate == self.patienceNumber:
                        SUCCESS = True
                    queueQ.pop(0) # Remove first element in queueQ from window for next variance computation
                    self.VAR_recon.append(VAR) # Store current variance to plot variance curve after

        else:
            #####################################  Exponential Moving Variance  #############################################
            if (not MV_metrics_already_stored_in_csv):
                # Compute variance for this window
                self.EMV = (1-self.alpha_EMV) * (self.EMV + self.alpha_EMV * norm(out_cropped - self.EMA)**2)
                # Compute EMA to be used in next window
                self.EMA = (1-self.alpha_EMV) * self.EMA + self.alpha_EMV * out_cropped
            else:
                self.EMV = MV_value_csv
            # Check if current variance is smaller than minimum previously computed variance, else count number of iterations since this minimum
            if self.EMV < VAR_min and not SUCCESS:
                VAR_min = self.EMV
                self.epochStar = epoch  # current detection point
                stagnate = 1
            else:
                stagnate += 1
            # ES point has been found
            if (stagnate == self.patienceNumber) and (self.epochStar > 0):
                SUCCESS = True
            if (not MV_metrics_already_stored_in_csv):
                self.VAR_recon.append(self.EMV) # Store current variance to plot variance curve after

        # Wait one iteration after SUCCESS to save ES point
        if self.SUCCESS:
            import matplotlib.pyplot as plt
            import numpy as np
            if (not self.SUCCESS):
                plt.plot(np.log(self.VAR_recon),label="Outer iteration : " + str(self.global_it))
                plt.legend()
                plt.ylabel("EMV (log scale)")
                plt.xlabel("DIP Iterations")
                plt.savefig(self.subroot_phantom + 'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/MV_global_' + str(self.global_it) + '.png')
            # Open output corresponding to epoch star
            net_output_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(self.epochStar) + '.img'
            # Open ckpt corresponding to epoch star
            ckpt_path = self.subroot_phantom+'Block2/' + self.suffix + '/checkpoint/' + format(self.experiment) + '/' + str(self.global_it) + '/epoch=' + format(self.epochStar) + '-step=' + format(self.epochStar) + '.ckpt'
            
            self.save_DIP_output(ckpt_path, net_output_path)
            
            out = self.fijii_np(net_output_path,shape=(self.PETImage_shape),type_im='<f')
            
            # Descale like at the beginning
            out = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #out = self.descale_imag(from_numpy(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)

            # Saving ES point image
            net_output_path = self.subroot_phantom + 'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/ES_out_' + self.net +  str(self.global_it) + '_epoch=' + format(self.epochStar) + '.img'
            self.save_img(out, net_output_path)
            print("#### MV ########################################################")
            print("                 ES point found, epoch* =", self.epochStar)
            print("#################################################################")
        
        if SUCCESS and (epoch != sub_iter_DIP - 1):
            self.SUCCESS = SUCCESS
        else:
            if (epoch == sub_iter_DIP - 1): # No ES was found, so set it to user defined value
                # self.epochStar = current_DIP_iteration - self.patienceNumber
                self.epochStar = self.DIP_it_if_no_ES_found - 1
                print(self.epochStar)
            
                # Open output corresponding to epoch star
                net_output_path = self.subroot_phantom+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(self.epochStar) + '.img'
                # Open ckpt corresponding to epoch star
                ckpt_path = self.subroot_phantom+'Block2/' + self.suffix + '/checkpoint/' + format(self.experiment) + '/' + str(self.global_it) + '/epoch=' + format(self.epochStar) + '-step=' + format(self.epochStar) + '.ckpt'
                
                self.save_DIP_output(ckpt_path, net_output_path)
            
        return SUCCESS, VAR_min, stagnate
    
    def initialize_MV(self,config,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,sub_iter_DIP,root, subroot, scanner, simulation, hyperparameters_list,image_net_input=None):          
        self.subroot = subroot
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.scaling_input = scaling_input
        self.suffix = suffix
        self.global_it = global_it
        self.scanner = scanner
        self.simulation = simulation
        self.image_net_input = image_net_input
        self.sub_iter_DIP = sub_iter_DIP
        self.hyperparameters_list = hyperparameters_list
        # Initialize variables
        self.do_everything(config,root)

    def run_MV(self,out,config, i, MV_metrics_already_stored_in_csv=False):
        if (self.DIP_early_stopping):

            if (config["read_only_MV_csv"]):
                MV_value_csv = self.VAR_recon[i]
            else:
                MV_value_csv = NaN
            self.SUCCESS,self.VAR_min,self.stagnate = self.compute_MV_value(out,i,self.sub_iter_DIP,self.queueQ,self.SUCCESS,self.VAR_min,self.stagnate,descale=False,MV_value_csv=MV_value_csv, MV_metrics_already_stored_in_csv=MV_metrics_already_stored_in_csv)
            if (not config["read_only_MV_csv"]):
                self.VAR_recon = self.VAR_recon
                self.MSE_MV = self.MSE_MV
                self.PSNR_MV = self.PSNR_MV
                self.SSIM_MV = self.SSIM_MV
            self.epochStar = self.epochStar
            self.patienceNumber = self.patienceNumber

            if self.SUCCESS: # Will be true 1 epoch after self.SUCCESS becomes True
                print("SUCCESS MVVVVVVVVVVVVVVVVVV")
            #     return 1
            # return 0

            if config["EMV_or_WMV"] == "EMV":
                self.alpha_EMV = self.alpha_EMV
            else:
                self.windowSize = self.windowSize
        
            return self.SUCCESS, self.VAR_recon, self.MSE_MV, self.PSNR_MV, self.SSIM_MV, self.epochStar, self.patienceNumber
        
    def save_DIP_output(self, ckpt_path, net_output_path):
        # Load ckpt file with pytorch ligthning and return the output of DIP network
        model = self.model_class.load_from_checkpoint(ckpt_path)
        # Get the output
        output = model(self.image_net_input_torch)
        image_net_output = squeeze(output.detach().numpy())
        # Save the output
        self.save_img(image_net_output, net_output_path)