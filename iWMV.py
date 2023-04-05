from torch import from_numpy
import numpy as np

# Local files to import
from vGeneral import vGeneral

class iWMV(vGeneral):
    def __init__(self, config):
        #super().__init__()

        ## Variables for WMV ##
        self.queueQ = []
        self.VAR_min = np.inf
        self.SUCCESS = False
        self.stagnate = 0

    def initializeSpecific(self,config,root, *args, **kwargs):
        self.patienceNumber = config["patienceNumber"]
        self.epochStar = -1
        self.VAR_recon = []
        self.MSE_WMV = []
        self.PSNR_WMV = []
        self.SSIM_WMV = []
        self.SUCCESS = False

        self.EMV_or_WMV = config["EMV_or_WMV"]
        if self.EMV_or_WMV == "EMV":    
            self.EMA = np.zeros((self.PETImage_shape))
            self.EMV = 0
            self.alpha_EMV = config["alpha_EMV"]
        else:
            self.windowSize = config["windowSize"]

        #self.queueQ = np.array((self.windowSize,self.PETImage_shape))

        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(np.float64)

    def runComputation(self,config,root):
        pass

    def WMV(self,out,epoch,queueQ,SUCCESS,VAR_min,stagnate):
        
        # Descale, squeeze image and add 3D dimension to 1 (ok for 2D images)
        out = self.descale_imag(from_numpy(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
        out = np.squeeze(out)
        if (len(out.shape) == 2): # 2D
            out = out[:,:,np.newaxis]
            phantom_ROI = self.get_phantom_ROI(self.phantom)
        else: # 3D
            #out = out.reshape(out.shape[::-1])
            phantom_ROI = np.ones_like(out)

        out_cropped = out * phantom_ROI
        image_gt_cropped = self.image_gt * phantom_ROI

        from skimage.metrics import peak_signal_noise_ratio
        from skimage.metrics import structural_similarity
        self.MSE_WMV.append(np.mean((image_gt_cropped - out_cropped)**2))
        self.PSNR_WMV.append(peak_signal_noise_ratio(image_gt_cropped, out_cropped, data_range=np.amax(out_cropped) - np.amin(out_cropped)))
        self.SSIM_WMV.append(structural_similarity(np.squeeze(image_gt_cropped), np.squeeze(out_cropped), data_range=out_cropped.max() - out_cropped.min()))

        if (self.EMV_or_WMV == "WMV"):
            #'''
            #####################################  Window Moving Variance  #############################################
            queueQ.append(out_cropped.flatten()) # Add last computed image to last element in queueQ from window
            if (len(queueQ) == self.windowSize):
                # Compute mean for this window
                mean = queueQ[0].copy()
                for x in queueQ[1:self.windowSize]:
                    mean += x
                mean = mean / self.windowSize
                # Compute variance for this window
                VAR = np.linalg.norm(queueQ[0] - mean) ** 2
                for x in queueQ[1:self.windowSize]:
                    VAR += np.linalg.norm(x - mean) ** 2
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
            #'''
        else:
            #'''
            #####################################  Exponential Moving Variance  #############################################
            # Compute variance for this window
            self.EMV = (1-self.alpha_EMV) * (self.EMV + self.alpha_EMV * np.linalg.norm(out_cropped - self.EMA)**2)
            # Compute EMA to be used in next window
            self.EMA = (1-self.alpha_EMV) * self.EMA + self.alpha_EMV * out_cropped
            # Check if current variance is smaller than minimum previously computed variance, else count number of iterations since this minimum
            if self.EMV < VAR_min and not SUCCESS:
                VAR_min = self.EMV
                self.epochStar = epoch  # current detection point
                stagnate = 1
            else:
                stagnate += 1
            # ES point has been found
            if stagnate == self.patienceNumber:
                SUCCESS = True
            self.VAR_recon.append(self.EMV) # Store current variance to plot variance curve after
            #'''


        #'''
        if SUCCESS:
            # Open output corresponding to epoch star
            net_outputs_path = self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + self.net + format(self.global_it) + '_epoch=' + format(self.epochStar) + '.img'
            out = self.fijii_np(net_outputs_path,shape=(self.PETImage_shape),type_im='<f')
            
            # Descale like at the beginning
            out = self.descale_imag(out,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
            #out = self.descale_imag(from_numpy(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)

            # Saving ES point image
            net_outputs_path = self.subroot + 'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/ES_out_' + self.net +  str(self.global_it) + '_epoch=' + format(self.epochStar) + '.img'
            self.save_img(out, net_outputs_path)
            print("#### WMV ########################################################")
            print("                 ES point found, epoch* =", self.epochStar)
            print("#################################################################")
        #'''

        return SUCCESS, VAR_min, stagnate