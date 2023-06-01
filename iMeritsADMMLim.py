## Python libraries
# Math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Useful


# Local files to import
from vGeneral import vGeneral

class iMeritsADMMLim(vGeneral):
    def __init__(self,config, *args, **kwargs):
        print("__init__")

    def initializeSpecific(self,config,root, *args, **kwargs):
        self.alpha = config["alpha"]
        self.nb_outer_iteration = config["nb_outer_iteration"]
        self.nb_inner_iteration = config["nb_inner_iteration"]
        #self.adaptive_parameters == config["adaptive_parameters"]

        self.bkg_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "background_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        self.hot_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "tumor_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        self.cold_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + self.phantom + '/' + "cold_mask" + self.phantom[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
        self.phantom_ROI = self.get_phantom_ROI(self.phantom)

        #Loading Ground Truth image to compute metrics
        self.image_gt = self.fijii_np(self.subroot_data + 'Data/database_v2/' + self.phantom + '/' + self.phantom + '.raw',shape=(self.PETImage_shape),type_im='<f')
        if config["FLTNB"] == "double":
            self.image_gt = self.image_gt.astype(np.float64)

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

    def runComputation(self,config,root):

        nbTuners = 1

        if self.tuners_tag == 'alphas':
            outer_iters = self.outers
            tuners = self.alpha

        elif self.tuners_tag == 'adaptiveRho':
            alpha0s = self.alpha
            duplicate = ''
            if self.REPLICATES:
                duplicate += '_rep' + str(self.replicate)
            outer_iters = self.outers
            tuners = alpha0s
            fp = open(self.subroot + self.suffix + '/adaptiveProcess' + str(duplicate) + '.log', mode='w+')
            fp = open(self.subroot + self.suffix + '/adaptiveProcess' + str(duplicate) + '.log', mode='w+')

        self.likelihoods_alpha = []

        if self.tuners_tag == 'alphas':
            # Extract likelihood from CASToR log file
            self.likelihoods = []
            if self.nb_inner_iteration == 1:
                logfile_name = '0.log'
            path_log = self.subroot + self.suffix + '/' + logfile_name
            self.extract_likelihood_from_log(path_log)

            self.PLOT(outer_iters, self.likelihoods, tuners, nbTuners, figNum=6,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Likelihood(same scale)',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)
            plt.ylim([2.904e6, 2.919e6])

            self.PLOT(outer_iters, self.likelihoods, tuners, nbTuners, figNum=1,
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
                coeff_alpha = self.getValueFromLogRow(path_txt, 0)/self.getValueFromLogRow(path_txt, 4)


                imagePath = self.subroot + self.suffix + '/' + imageName
                IR, MSE, CRC, MA = self.computeThose4(imagePath)
                IR_bkgs.append(IR)
                MSEs.append(MSE)
                CRC_hots.append(CRC)
                MA_colds.append(MA)

                Xnorms.append(self.computeNorm(imagePath))
                #Vnorms.append(self.computeNorm(self.subroot + self.suffix + '/'+vName))
                #u_norm = self.computeNorm(self.subroot + self.suffix + '/'+uName)
                #Unorms.append(u_norm)
                #U_unscaled_norms.append(u_norm * coeff_alpha)
                coeff_alphas.append(coeff_alpha)
                #averageUs.append(computeAverage(self.subroot + self.suffix + '/'+uName))

            self.PLOT(outer_iters, IR_bkgs, tuners, nbTuners, figNum=2,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Image Roughness in the background',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, MSEs, tuners, nbTuners, figNum=3,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Mean Square Error',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, CRC_hots, tuners, nbTuners, figNum=4,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='CRC hot',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, MA_colds, tuners, nbTuners, figNum=5,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='MA cold',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, Xnorms, tuners, nbTuners, figNum=7,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='norm of x',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            '''
            self.PLOT(outer_iters, Vnorms, tuners, nbTuners, figNum=8,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='norm of v',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, Unorms, tuners, nbTuners, figNum=9,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='norm of u',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, U_unscaled_norms, tuners, nbTuners, figNum=10,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='norm of UNSCALED u',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)
            
            self.PLOT(outer_iters, averageUs, tuners, nbTuners, figNum=12,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='average of u',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)
            '''
            self.PLOT(outer_iters, coeff_alphas, tuners, nbTuners, figNum=11,
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
                adaptiveAlphas.append(self.getValueFromLogRow(path_txt, 0))

                # get adaptive tau
                adaptiveTaus.append(self.getValueFromLogRow(path_txt, 2))

                # get relative primal residual
                relPrimals.append(self.getValueFromLogRow(path_txt, 6))

                # get relative dual residual
                relDuals.append(self.getValueFromLogRow(path_txt, 8))

                # get xi
                xis.append(self.getValueFromLogRow(path_txt, 6) / (self.getValueFromLogRow(path_txt, 8) * 2))

                if self._3NORMS:
                    # get norm of Ax(n+1) - v(n+1)
                    normAxvs.append(self.getValueFromLogRow(path_txt, 10))

                    # get norm of Ax(n+1) - v(n) + u(n)
                    normAxvus.append(self.getValueFromLogRow(path_txt, 12))

                    # get norm of Ax(n+1) - v(n+1) + u(n)
                    normAxv1us.append(self.getValueFromLogRow(path_txt, 14))

                if self._2R:
                    # get norm of primal residual
                    primals.append(self.getValueFromLogRow(path_txt, 16))

                    # get norm of dual residual
                    duals.append(self.getValueFromLogRow(path_txt, 18))

            self.PLOT(outer_iters, adaptiveAlphas, tuners, nbTuners, figNum=1,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Adaptive alpha',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, adaptiveTaus, tuners, nbTuners, figNum=2,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Adaptive taus',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, relPrimals, tuners, nbTuners, figNum=3,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Relative primal residuals',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, relDuals, tuners, nbTuners, figNum=4,
                Xlabel='Outer iteration',
                Ylabel='The legend shows different alpha',
                Title='Relative dual residuals',
                replicate=self.replicate,
                whichOptimizer=self.whichADMMoptimizer,
                imagePath=self.fomSavingPath)

            self.PLOT(outer_iters, xis, tuners, nbTuners, figNum=5,
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
                self.PLOT(outer_iters, normAxvs, tuners, nbTuners, figNum=6,
                    Xlabel='Outer iteration',
                    Ylabel='The legend shows different alpha',
                    Title='norm of Ax(n+1) - v(n+1)',
                    replicate=self.replicate,
                    whichOptimizer=self.whichADMMoptimizer,
                    imagePath=self.fomSavingPath)

                self.PLOT(outer_iters, normAxvus, tuners, nbTuners, figNum=7,
                    Xlabel='Outer iteration',
                    Ylabel='The legend shows different alpha',
                    Title='norm of Ax(n+1) - v(n) + u(n)',
                    replicate=self.replicate,
                    whichOptimizer=self.whichADMMoptimizer,
                    imagePath=self.fomSavingPath)

                self.PLOT(outer_iters, normAxv1us, tuners, nbTuners, figNum=8,
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
                self.PLOT(outer_iters, primals, tuners, nbTuners, figNum=9,
                    Xlabel='Outer iteration',
                    Ylabel='The legend shows different alpha',
                    Title='primal residual',
                    replicate=self.replicate,
                    whichOptimizer=self.whichADMMoptimizer,
                    imagePath=self.fomSavingPath)

                self.PLOT(outer_iters, duals, tuners, nbTuners, figNum=10,
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
        elif self.tuners_tag == 'alphas' and len(self.alpha)==len(self.likelihoods_alpha):
            plt.figure()
            plt.plot(self.alpha, self.likelihoods_alpha, '-x')
            plt.xlabel('alpha')
            plt.title('likelihood')

        if self.SHOW:
            plt.show()
        '''

    def PLOT(self,
            X,
            Y,
            tuners,
            nbTuner,
            figNum=1,
            Xlabel='X',
            Ylabel='Y',
            Title='',
            beginning=0,
            bestValue=-1,
            showLess=[],
            replicate=0,
            imagePath='',
            whichOptimizer='',
            Together=True):

        plt.figure(figNum)
        end = len(X)
        
        if nbTuner < 10:
            plt.plot(X[beginning:end], Y[beginning:end], label=str(tuners))
        elif 10 <= nbTuner < 20:
            plt.plot(X[beginning:end], Y[beginning:end], '.-', label=str(tuners))
        else:
            plt.plot(X[beginning:end], Y[beginning:end], 'x-', label=str(tuners))
        plt.legend(loc='best')
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.title('(' + whichOptimizer + ')(replicate ' + str(replicate) + ') ' + Title)
        if 'same scale' in Title:
            plt.ylim([2.904e6, 2.919e6])
        if Together:
            if replicate > 0 and tuners == tuners:
                self.mkdir(imagePath)
                plt.savefig(imagePath + '/(' + whichOptimizer + ')' + Title + '_rep' + str(replicate) + '.png')
        elif not Together:
            self.mkdir(imagePath)
            plt.savefig(imagePath + '/(' + whichOptimizer + ')' + Title + '_rep' + str(replicate) + ' - ' + str(tuners) + '.png')


    def getValueFromLogRow(self,pathLog, row):
        log = pd.read_table(pathLog)
        theRow = log.loc[[row]]
        theRowArray = np.array(theRow)
        theRowString = theRowArray[0, 0]
        theValue = float(theRowString)

        return theValue


    def computeNorm(self, f_path, type_im='<d'):
        dtype = np.dtype(type_im)
        fid = open(f_path, 'rb')
        data = np.fromfile(fid, dtype)
        data_norm = np.linalg.norm(data)

        return data_norm

    #'''
    def computeThose4(self,f,image='image0'):
        # have the image f as input, return IR_bkg_recon, MSE_recon, CRC_hot_recon, MA_cold_recon
        f = self.fijii_np(f, shape=self.PETImage_shape, type='<d')
        bkg_ROI_act = f[self.bkg_ROI == 1]
        # IR
        if np.mean(bkg_ROI_act) != 0:
            IR_bkg_recon = np.std(bkg_ROI_act) / np.mean(bkg_ROI_act)
        else:
            IR_bkg_recon = 1e10

        # MSE
        MSE_recon = np.mean((self.image_gt * self.bkg_ROI - f * self.bkg_ROI) ** 2)

        # Mean Concentration Recovery coefficient (CRCmean) in hot cylinder calculation (-c 50. 10. 0. 20. 4. 400)
        hot_ROI_act = f[self.hot_ROI == 1]

        # CRC hot
        # CRC_hot_recon.append(np.mean(hot_ROI_act) / 400.)
        CRC_hot_recon = np.mean(hot_ROI_act) - 400.
        
        cold_ROI_act = f[self.cold_ROI == 1]

        # MA cold
        MA_cold_recon = np.mean(cold_ROI_act)

        return IR_bkg_recon, MSE_recon, CRC_hot_recon, MA_cold_recon
    #'''    

    '''
    def computeThose4(self):
        from csv import reader as reader_csv
        import numpy as np

        PSNR_recon = []
        PSNR_norm_recon = []
        MSE_recon = []
        SSIM_recon = []
        MA_cold_recon = []
        AR_hot_recon = []
        AR_bkg_recon = []
        IR_bkg_recon = []

        # Load metrics from csv file
        metrics_file = self.subroot_metrics + self.method + '/' + self.suffix + '/' + 'metrics.csv'
        with open(metrics_file, 'r') as myfile:
            spamreader = reader_csv(myfile,delimiter=';')
            rows_csv = list(spamreader)
            rows_csv[0] = [float(i) for i in rows_csv[0]]
            rows_csv[1] = [float(i) for i in rows_csv[1]]
            rows_csv[2] = [float(i) for i in rows_csv[2]]
            rows_csv[3] = [float(i) for i in rows_csv[3]]
            rows_csv[4] = [float(i) for i in rows_csv[4]]
            rows_csv[5] = [float(i) for i in rows_csv[5]]
            rows_csv[6] = [float(i) for i in rows_csv[6]]
            rows_csv[7] = [float(i) for i in rows_csv[7]]

            PSNR_recon.append(np.array(rows_csv[0]))
            PSNR_norm_recon.append(np.array(rows_csv[1]))
            MSE_recon.append(np.array(rows_csv[2]))
            SSIM_recon.append(np.array(rows_csv[3]))
            MA_cold_recon.append(np.array(rows_csv[4]))
            AR_hot_recon.append(np.array(rows_csv[5]))
            AR_bkg_recon.append(np.array(rows_csv[6]))
            IR_bkg_recon.append(np.array(rows_csv[7]))
    '''