from show_functions import computeThose4, PLOT
from show_functions import getValueFromLogRow, computeNorm, computeAverage

## Python libraries
# Math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Useful


# Local files to import
#from vGeneral import vGeneral
from iResults import iResults
from vGeneral import vGeneral

class iMeritsADMMLim(vGeneral):
    def __init__(self,config):
        print("__init__")

    def initializeSpecific(self,settings_config,fixed_config,hyperparameters_config,root):
        self.alpha = hyperparameters_config["alpha"]
        self.nb_outer_iteration = hyperparameters_config["nb_outer_iteration"]
        self.nb_inner_iteration = fixed_config["nb_inner_iteration"]
        #self.adaptive_parameters == hyperparameters_config["adaptive_parameters"]

        #inners = list(range(innerIteration))
        self.outers = list(range(1,self.nb_outer_iteration+1))
        self.REPLICATES = True  # as we use variable 'replicates' above, set it to True
        self._3NORMS = True  # defaut:True
        self._2R = True  # defaut:True
        self.fomSavingPath = self.subroot + 'Images/tmp/'

        option = 1  # Now, only option 0 and 1 are useful, option 2, 3 and 4 should be ignored
        #            0            1              2              3                 4
        OPTION = ['alphas', 'adaptiveRho', 'inner_iters', 'outer_iters', 'calculateDiffCurve']
        self.tuners_tag = OPTION[option]
        
        self._squreNorm = False  # defaut:False
        self.whichADMMoptimizer = 'ADMMLim'
        self.SHOW = False  # show the plots in python or not !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def runComputation(self,config,settings_config,fixed_config,hyperparameters_config,root):

        nbTuners = 1

        if self.tuners_tag == 'alphas':
            outer_iters = self.outers
            tuners = self.alpha

        elif self.tuners_tag == 'adaptiveRho':
            # self.nb_outer_iteration = 1000
            # self.nb_inner_iteration = 50
            alpha0s = self.alpha
            duplicate = ''
            if self.REPLICATES:
                duplicate += '_rep' + str(self.replicate)
            inner_iters = range(self.nb_inner_iteration)
            outer_iters = self.outers
            # dataFolderPath = 'ADMM-old-adaptive+i50+o70+alpha0=...*16+3+2'
            tuners = alpha0s
            fp = open(self.subroot + self.suffix + '/adaptiveProcess' + str(duplicate) + '.log', mode='w+')
            fp = open(self.subroot + self.suffix + '/adaptiveProcess' + str(duplicate) + '.log', mode='w+')

        likelihoods_alpha = []

        if self.tuners_tag == 'alphas':
            likelihoods = []
            for outer_iter in range(self.nb_outer_iteration):
                if self.REPLICATES:
                    replicatesPath = '/replicate_' + str(self.replicate) + '/' + self.whichADMMoptimizer \
                                    # + '/Comparison/' + self.whichADMMoptimizer
                else:
                    replicatesPath = ''
                if self.nb_inner_iteration == 1:
                    logfile_name = '0.log'
                path_log = self.subroot + self.suffix + '/' + logfile_name
                theLog = pd.read_table(path_log)
                #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                #    print(theLog)
                #import re
                #likelihoodRows = [re.findall('Log-likelihood:',line) for line in theLog]
                

                fileRows = np.column_stack([theLog[col].str.contains("Log-likelihood", na=False) for col in theLog])
                
                likelihoodRows = np.array(theLog.loc[fileRows == 1])
                for rows in likelihoodRows:
                    theLikelihoodRowString = rows[0][22:44]
                if theLikelihoodRowString[0] == '-':
                    theLikelihoodRowString = '0'
                likelihood = float(theLikelihoodRowString)
                likelihoods_alpha.append(likelihood)
                likelihoods.append(likelihood)

            PLOT(outer_iters, likelihoods, tuners, nbTuners, figNum=6,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Likelihood(same scale)',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)
            plt.ylim([2.904e6, 2.919e6])

            PLOT(outer_iters, likelihoods, tuners, nbTuners, figNum=1,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Likelihood',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            IR_bkgs = []
            MSEs = []
            CRC_hots = []
            MA_colds = []
            Xnorms = []
            Vnorms = []
            Unorms = []
            U_unscaled_norms = []
            coeff_alphas = []
            averageUs = []
            for outer_iter in range(1,self.nb_outer_iteration+1):
                if self.REPLICATES:
                    replicatesPath = '/replicate_' + str(self.replicate) + '/' + self.whichADMMoptimizer \
                                    #+ '/Comparison/' + self.whichADMMoptimizer
                else:
                    replicatesPath = ''
                imageName = '0_it' + str(outer_iter) + '.img'
                #vName = '0_' + str(outer_iter) + '_v.img'
                #uName = '0_' + str(outer_iter) + '_u.img'

                logfile_name = '0_adaptive_it' + str(outer_iter) + '.log'
                path_txt = self.subroot + self.suffix + '/' + logfile_name
                coeff_alpha = getValueFromLogRow(path_txt, 0)/getValueFromLogRow(path_txt, 4)



                imagePath = self.subroot + self.suffix + '/' + imageName
                IR, MSE, CRC, MA = computeThose4(imagePath)
                IR_bkgs.append(IR)
                MSEs.append(MSE)
                CRC_hots.append(CRC)
                MA_colds.append(MA)

                Xnorms.append(computeNorm(imagePath))
                #Vnorms.append(computeNorm(self.subroot + self.suffix + '/'+vName))
                #u_norm = computeNorm(self.subroot + self.suffix + '/'+uName)
                #Unorms.append(u_norm)
                #U_unscaled_norms.append(u_norm * coeff_alpha)
                coeff_alphas.append(coeff_alpha)
                #averageUs.append(computeAverage(self.subroot + self.suffix + '/'+uName))

            PLOT(outer_iters, IR_bkgs, tuners, nbTuners, figNum=2,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Image Roughness in the background',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, MSEs, tuners, nbTuners, figNum=3,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Mean Square Error',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, CRC_hots, tuners, nbTuners, figNum=4,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='CRC hot',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, MA_colds, tuners, nbTuners, figNum=5,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='MA cold',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, Xnorms, tuners, nbTuners, figNum=7,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='norm of x',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            '''
            PLOT(outer_iters, Vnorms, tuners, nbTuners, figNum=8,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='norm of v',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, Unorms, tuners, nbTuners, figNum=9,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='norm of u',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, U_unscaled_norms, tuners, nbTuners, figNum=10,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='norm of UNSCALED u',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)
            
            PLOT(outer_iters, averageUs, tuners, nbTuners, figNum=12,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='average of u',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)
            '''
            PLOT(outer_iters, coeff_alphas, tuners, nbTuners, figNum=11,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='coeff_alphas',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

        # adaptive self.alpha
        elif self.tuners_tag == 'adaptiveRho':
            adaptiveAlphas = []
            adaptiveTaus = []
            relPrimals = []
            relDuals = []
            xis = []
            normAxvs = []
            normAxvus = []
            normAxv1us = []
            primals = []
            duals = []
            for outer_iter in range(1,self.nb_outer_iteration+1):
                if self.REPLICATES:
                    replicatesPath = '/replicate_' + str(self.replicate) + '/' + self.whichADMMoptimizer \
                                    #+ '/Comparison/' + self.whichADMMoptimizer
                else:
                    replicatesPath = ''
                logfile_name = '0_adaptive_it' + str(outer_iter) + '.log'
                path_txt = self.subroot + self.suffix + '/' + logfile_name

                # get adaptive alpha
                adaptiveAlphas.append(getValueFromLogRow(path_txt, 0))

                # get adaptive tau
                adaptiveTaus.append(getValueFromLogRow(path_txt, 2))

                # get relative primal residual
                relPrimals.append(getValueFromLogRow(path_txt, 6))

                # get relative dual residual
                relDuals.append(getValueFromLogRow(path_txt, 8))

                # get xi
                xis.append(getValueFromLogRow(path_txt, 6) / (getValueFromLogRow(path_txt, 8) * 2))

                if self._3NORMS:
                    # get norm of Ax(n+1) - v(n+1)
                    normAxvs.append(getValueFromLogRow(path_txt, 10))

                    # get norm of Ax(n+1) - v(n) + u(n)
                    normAxvus.append(getValueFromLogRow(path_txt, 12))

                    # get norm of Ax(n+1) - v(n+1) + u(n)
                    normAxv1us.append(getValueFromLogRow(path_txt, 14))

                if self._2R:
                    # get norm of primal residual
                    primals.append(getValueFromLogRow(path_txt, 16))

                    # get norm of dual residual
                    duals.append(getValueFromLogRow(path_txt, 18))

            PLOT(outer_iters, adaptiveAlphas, tuners, nbTuners, figNum=1,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Adaptive self.alpha',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, adaptiveTaus, tuners, nbTuners, figNum=2,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Adaptive taus',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, relPrimals, tuners, nbTuners, figNum=3,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Relative primal residuals',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, relDuals, tuners, nbTuners, figNum=4,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Relative dual residuals',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            PLOT(outer_iters, xis, tuners, nbTuners, figNum=5,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Xis',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            if self._3NORMS:
                if self._squreNorm:
                    normAxvs = np.sqrt(normAxvs)
                    normAxvus = np.sqrt(normAxvus)
                    normAxv1us = np.sqrt(normAxv1us)
                PLOT(outer_iters, normAxvs, tuners, nbTuners, figNum=6,
                    Xlabel='Outer iteration',
                    Ylabel='The legend shows different alpha',
                    Title='norm of Ax(n+1) - v(n+1)',
                    replicate=self.replicate,
                    whichOptimizer=self.whichADMMoptimizer,
                    imagePath=self.fomSavingPath)

                PLOT(outer_iters, normAxvus, tuners, nbTuners, figNum=7,
                    Xlabel='Outer iteration',
                    Ylabel='The legend shows different alpha',
                    Title='norm of Ax(n+1) - v(n) + u(n)',
                    replicate=self.replicate,
                    whichOptimizer=self.whichADMMoptimizer,
                    imagePath=self.fomSavingPath)

                PLOT(outer_iters, normAxv1us, tuners, nbTuners, figNum=8,
                    Xlabel='Outer iteration',
                    Ylabel='The legend shows different alpha',
                    Title='norm of Ax(n+1) - v(n+1) + u(n)',
                    replicate=self.replicate,
                    whichOptimizer=self.whichADMMoptimizer,
                    imagePath=self.fomSavingPath)

            if self._2R:
                if self._squreNorm:
                    primals = np.sqrt(primals)
                    duals = np.sqrt(duals)
                PLOT(outer_iters, primals, tuners, nbTuners, figNum=9,
                    Xlabel='Outer iteration',
                    Ylabel='The legend shows different alpha',
                    Title='primal residual',
                    replicate=self.replicate,
                    whichOptimizer=self.whichADMMoptimizer,
                    imagePath=self.fomSavingPath)

                PLOT(outer_iters, duals, tuners, nbTuners, figNum=10,
                    Xlabel='Outer iteration',
                    Ylabel='The legend shows different alpha',
                    Title='dual residual',
                    replicate=self.replicate,
                    whichOptimizer=self.whichADMMoptimizer,
                    imagePath=self.fomSavingPath)

            print('No.' + str(nbTuners), '  initial alpha =', tuners, '\trel primal', '\trel dual', file=fp)
            print(file=fp)
            for k in range(1, len(adaptiveAlphas) + 1):
                if k < 10:
                    print('           --(   ' + str(k) + ')-->', adaptiveAlphas[k - 1], '\t', relPrimals[k - 1], '\t',
                        relDuals[k - 1], file=fp)
                elif k < 100:
                    print('           --(  ' + str(k) + ')-->', adaptiveAlphas[k - 1], '\t', relPrimals[k - 1], '\t',
                        relDuals[k - 1], file=fp)
                elif k < 1000:
                    print('           --( ' + str(k) + ')-->', adaptiveAlphas[k - 1], '\t', relPrimals[k - 1], '\t',
                        relDuals[k - 1], file=fp)
                else:
                    print('           --(' + str(k) + ')-->', adaptiveAlphas[k - 1], '\t', relPrimals[k - 1], '\t',
                        relDuals[k - 1], file=fp)
            print(file=fp)
            print(file=fp)

        if self.tuners_tag == 'adaptiveRho':
            fp.close()

        '''
        elif self.tuners_tag == 'alphas' and len(self.alpha)==len(likelihoods_alpha):
            plt.figure()
            plt.plot(self.alpha, likelihoods_alpha, '-x')
            plt.xlabel('alpha')
            plt.title('likelihood')

        if self.SHOW:
            plt.show()
        '''