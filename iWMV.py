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
        self.windowSize = config["windowSize"]
        self.patienceNumber = config["patienceNumber"]
        self.epochStar = -1
        self.VAR_recon = []
        self.MSE_WMV = []
        self.PSNR_WMV = []
        self.SSIM_WMV = []
        self.SUCCESS = False

        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type='<f')
        if config["FLTNB"] == "double":
            self.image_gt.astype(np.float64)

    def runComputation(self,config,root):
        pass

    def WMV(self,out,epoch,queueQ,SUCCESS,VAR_min,stagnate):
        
        # Descale, squeeze image and add 3D dimension to 1 (ok for 2D images)
        out = self.descale_imag(from_numpy(out),self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input)
        out = np.squeeze(out)
        out = out[:,:,np.newaxis]

        phantom_ROI = self.get_phantom_ROI()
        out_cropped = out * phantom_ROI
        image_gt_cropped = self.image_gt * phantom_ROI

        from skimage.metrics import peak_signal_noise_ratio
        from skimage.metrics import structural_similarity
        self.MSE_WMV.append(np.mean((image_gt_cropped - out_cropped)**2))
        self.PSNR_WMV.append(peak_signal_noise_ratio(image_gt_cropped, out_cropped, data_range=np.amax(out_cropped) - np.amin(out_cropped)))
        self.SSIM_WMV.append(structural_similarity(np.squeeze(image_gt_cropped), np.squeeze(out_cropped), data_range=out_cropped.max() - out_cropped.min()))

        #####################################  Window Moving Variance  #############################################
        queueQ.append(out_cropped.flatten())
        if (len(queueQ) == self.windowSize):
            mean = queueQ[0].copy()
            for x in queueQ[1:self.windowSize]:
                mean += x
            mean = mean / self.windowSize
            VAR = np.linalg.norm(queueQ[0] - mean) ** 2
            for x in queueQ[1:self.windowSize]:
                VAR += np.linalg.norm(x - mean) ** 2
            VAR = VAR / self.windowSize
            if VAR < VAR_min and not SUCCESS:
                VAR_min = VAR
                self.epochStar = epoch  # detection point
                stagnate = 1
            else:
                stagnate += 1
            if stagnate == self.patienceNumber:
                SUCCESS = True
            queueQ.pop(0)
            self.VAR_recon.append(VAR)

        #'''
        if SUCCESS:
            # Saving ES point image
            net_outputs_path = self.subroot + 'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/ES_out_' + self.net + '_epoch=' + format(self.epochStar) + '.img'
            self.save_img(out, net_outputs_path)
            print("#### WMV ########################################################")
            print("                 ES point found, epoch* =", self.epochStar)
            print("#################################################################")
        #'''

        return SUCCESS, VAR_min, stagnate