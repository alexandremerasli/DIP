# Useful
from pathlib import Path
import os
import numpy as np
from pandas import read_table
import matplotlib.pyplot as plt

# Local files to import
from vGeneral import vGeneral

class iComputeLikelihood_nested(vGeneral):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        # Specific hyperparameters for reconstruction module (Do it here to have raytune config hyperparameters selection)
        if (config["method"] != "MLEM" and config["method"] != "OSEM" and config["method"] != "AML" and config["method"] != "OPTITR"):
            self.rho = config["rho"]
        else:
            self.rho = 0
        if ('ADMMLim' in config["method"] or  'nested' in config["method"] or  'Gong' in config["method"]):
            if (config["method"] != "ADMMLim"):
                self.unnested_1st_global_iter = config["unnested_1st_global_iter"]
            else:
                self.unnested_1st_global_iter = None
            if ( 'Gong' in config["method"]):
                self.alpha = None
            else:
                if (config["recoInNested"] == "ADMMLim"):
                    self.stoppingCriterionValue = config["stoppingCriterionValue"]
                    self.saveSinogramsUAndV = config["saveSinogramsUAndV"]
                    self.alpha = config["alpha"]
                    self.adaptive_parameters = config["adaptive_parameters"]
                else:
                    self.alpha = None
                    self.adaptive_parameters = "nothing"
                    self.A_AML = config["A_AML"]
                if (self.adaptive_parameters == "nothing"): # set mu, tau, xi to any values, there will not be used in CASToR
                    self.mu_adaptive = np.NaN
                    self.tau = np.NaN
                    self.xi = np.NaN
                    self.tau_max = np.NaN
                else:
                    self.mu_adaptive = config["mu_adaptive"]
                    self.tau = config["tau"]
                    self.xi = config["xi"]
                    if (self.adaptive_parameters == "both"):
                        self.tau_max = config["tau_max"]
                    else:
                        self.tau_max = np.NaN
        # Initialization
        self.recoInNested = config["recoInNested"]

        
    def runComputation(self,config,root):

        if (self.method == 'AML' or self.method == 'APGMAP'):
            self.A_AML = config["A_AML"]
        if (self.method == 'AML'):
            self.beta = config["A_AML"]
        elif ('ADMMLim' in self.method):
            self.beta = config["alpha"]
            self.recoInNested = "ADMMLim"
        elif (self.method == 'BSREM' or self.method == 'APGMAP'):
            self.beta = self.rho

        if (self.method != 'BSREM' and self.method != 'nested' and self.method != 'Gong' and self.method != 'APGMAP'):
            self.post_smoothing = config["post_smoothing"]
        else:
            self.post_smoothing = 0

    
        if 'nested' in config["method"] or 'Gong' in config["method"]:
            folder_sub_path = self.subroot + 'Block2/' + self.suffix
        else:
            folder_sub_path = self.subroot + '/' + self.suffix
        Path(folder_sub_path).mkdir(parents=True, exist_ok=True) # CASToR path
        
        config["castor_foms"] = True
        # config["method"] = "MLEM"
        self.method = "MLEM"
        
        likelihoods = []
        for i in range(self.max_iter):
            if 'nested' in config["method"] or 'Gong' in config["method"]:
                output_path = ' -fout ' + folder_sub_path + '/' + config["method"] + "_" + str(i-1) # Output path for CASTOR framework
                initialimage = ' -img ' + self.subroot + '/Block2/' + self.suffix + '/out_cnn/' + str(self.experiment) + '/out_' + self.net + str(i-1) + '_FINAL.hdr'
            else:
                output_path = ' -fout ' + folder_sub_path + '/' + config["method"] + "_" + str(i) # Output path for CASTOR framework
                initialimage = ' -img ' + self.subroot + '/' + self.suffix + '/' + config["method"] + '_it' + str(i) + '.hdr'
            it = ' -it 1:1'
        
            print("CASToR command line : ")
            castor_command_line = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str, self.phantom, self.replicate, self.post_smoothing) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho) + it + output_path + initialimage
            # Do not save images, only log file to retrieve likelihood
            castor_command_line += " -oit 10:10"
            print(castor_command_line)
            os.system(castor_command_line)
            

            if 'nested' in config["method"] or 'Gong' in config["method"]:
               logfile_name = config["method"] + '_' + str(i-1) + '.log'
            else:
                logfile_name = config["method"] + '_' + str(i) + '.log'
            path_log = folder_sub_path + '/' + logfile_name
            theLog = read_table(path_log)

            fileRows = np.column_stack([theLog[col].str.contains("Log-likelihood", na=False) for col in theLog])
            likelihoodRows = np.array(theLog.loc[fileRows == 1])
            for rows in likelihoodRows:
                theLikelihoodRowString = rows[0][22:44]
                if theLikelihoodRowString[0] == '-':
                    theLikelihoodRowString = '0'
                likelihood = float(theLikelihoodRowString)
                # likelihoods_alpha.append(likelihood)
                likelihoods.append(likelihood)

        if 'nested' in config["method"] or 'Gong' in config["method"]:
            plt.plot(np.arange(-1,self.max_iter-1),likelihoods)
        else:
            plt.plot(np.arange(0,self.max_iter-1),likelihoods)
        plt.xlabel("iterations")
        plt.ylabel("log-likelihood")
        plt.show()
        
        # # Initializing results class
        # if ((config["average_replicates"] and self.replicate == 1) or (config["average_replicates"] == False)):
        #     from iResults import iResults
        #     classResults = iResults(config)
        #     classResults.nb_replicates = self.nb_replicates
        #     classResults.debug = self.debug
        #     classResults.rho = self.rho
        #     classResults.hyperparameters_list = self.hyperparameters_list
        #     classResults.phantom_ROI = self.phantom_ROI
        #     if ("3D" not in self.phantom):
        #         classResults.bkg_ROI = self.bkg_ROI
        #         classResults.hot_TEP_ROI = self.hot_TEP_ROI
        #         classResults.hot_TEP_match_square_ROI = self.hot_TEP_match_square_ROI
        #         classResults.hot_perfect_match_ROI = self.hot_perfect_match_ROI
        #         classResults.hot_ROI = self.hot_ROI
        #         classResults.cold_ROI = self.cold_ROI
        #     classResults.initializeSpecific(config,root)
        #     classResults.runComputation(config,root)