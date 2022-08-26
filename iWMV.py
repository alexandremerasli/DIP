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

    def initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root):
        self.windowSize = fixed_config["windowSize"]
        self.patienceNumber = fixed_config["patienceNumber"]
        self.epochStar = -1
        self.VAR_recon = []


    def runComputation(self,config,settings_config,fixed_config,hyperparameters_config,root):
        pass

    def WMV(self,out,epoch,queueQ,SUCCESS,VAR_min,stagnate):
        #####################################  Window Moving Variance  #############################################
        x_out = out * self.get_phantom_ROI()
        queueQ.append(x_out.flatten())
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
            net_outputs_path = self.subroot + 'Block2/out_cnn/' + format(self.experiment) + '/ES_out_' + self.net + '_epoch=' + format(self.epochStar) + self.suffix + '.img'
            self.save_img(out, net_outputs_path)
            print("#### WMV ########################################################")
            print("                 ES point found, epoch* =", self.epochStar)
            print("#################################################################")
        #'''

        return SUCCESS, VAR_min, stagnate