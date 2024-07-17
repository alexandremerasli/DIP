from ray import tune

def config_func_MIC():

    # Configuration dictionary for general settings parameters
    # Parameters in this dictionary are not added in suffix of the output folder as they are not hyperparameters
    settings_config = {
        "image" : tune.grid_search(['image40_0']), # Image from database (data/Algo/Data/database_v2)
        "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
        "method" : tune.grid_search(["DNA"]), # Reconstruction algorithm (DNA, DIPRecon, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
        "processing_unit" : tune.grid_search(['CPU']), # Run NN training with pytorch on CPU or GPU
        "nb_threads" : tune.grid_search([1]), # Number of desired threads in reconstruction with CASToR. 0 means all the available threads will be used
        "FLTNB" : tune.grid_search(['float']), # FLTNB precision must be set as in CASToR. Default is float (meaning float32 in numpy)
        "ray" : True, # If True, run computation with raytune parallel computation (for several settings in parallel). Set it to False to debug code
        "tensorboard" : True, # If True, show results (images, metrics) in tensorboard during or after reconstruction. Set it to False to save time
        "all_images_DIP" : tune.grid_search(['False']), # Option to store only 10 images like in tensorboard (quicker, for visualization, set it to "True" by default). Can be set to "True", "False", "Unique" (store only last image)
        "experiment" : tune.grid_search([24]),
        #"replicates" : tune.grid_search([24]),
        #"replicates" : tune.grid_search([2,12,13,19,24,25,34,36]), # List of desired replicates to work with in parallel. list(range(1,n+1)) means n replicates
        "replicates" : tune.grid_search(list(range(1,40+1))), # List of desired replicates to work with in parallel. list(range(1,n+1)) means n replicates
        "castor_foms" : tune.grid_search([True]), # Set to True to compute CASToR Figure Of Merits or residuals for ADMMReg. Must be set to False with list mode data
    }
    # Configuration dictionary for some hyperparameters, usually fixed
    # Parameters in this dictionary are not added in suffix of the output folder
    fixed_config = {
        "max_iter" : tune.grid_search([100]), # Number of iterations for usual optimizers (MLEM, BSREM, AML etc.) and outer iterations for DNA and DIPRecon
        "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMReg)
        "use_u_and_v_DNA" : tune.grid_search([False]), # For DNA reconstruction, set to True to initialize current sinograms u and v by those from previous outer iteration
        "finetuning" : tune.grid_search(['last']),
        "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms (MRF)
        "sub_iter_DIP_init" : tune.grid_search([1000]), # Number of DIP iterations at DNA/DIPRecon initialization. Could be overrided if early stopping point is reached using "DIP_early_stopping_when" parameter
        "nb_inner_sub_iteration" : tune.grid_search([1]), # Number of inner subiterations in DNA (number of iterations of gradient descent in ADMM-Reg (if mlem_sequence is False). It should be 1 as it is coded for now in CASToR
        "xi" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in ADMMReg
        "net" : tune.grid_search(['DIP']), # Neural Network (NN) architecture to use ("DIP" (U-Net from DIPRecon paper), "DD" (Deep Decoder"), "DD_AE" (DD based autoencoder), "DIP_VAE" (DIP-based Variational AutoEncoder)))
        "DIP_early_stopping_when" : tune.grid_search(["never"]), # Use DIP early stopping - ES ("never" means no ES, "init" means ES only at initialization, "all" means ES at each iteration)
        "EMV_or_WMV" : tune.grid_search(["EMV"]), # WMV or EMV for DIP early stopping
        "alpha_EMV" : tune.grid_search([0.1]), # EMV forgetting factor alpha
        "windowSize" : tune.grid_search([50]), # WMV window size
        "patienceNumber" : tune.grid_search([100]), # Patience number in moving variance algorithms
    }
    # Configuration dictionary for hyperparameters to tune
    # Parameters in this dictionary are the only ones added in suffix of the output folder
    hyperparameters_config = {
        "image_init_path_without_extension" : tune.grid_search(['BSREM_it30']), # Initial image of the reconstruction algorithm (taken from subroot + "/Data/initialization")
        "rho" : tune.grid_search([0.003,8e-4,0.008,0.03]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (DNA and DIPRecon)
        "rho" : tune.grid_search([0.3,0.03,0.003,0.0003,0.00003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (DNA and DIPRecon)
        "rho" : tune.grid_search([0.003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (DNA and DIPRecon)
        #"rho" : tune.grid_search([0.0003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (DNA and DIPRecon)
        "adaptive_parameters_DIP" : tune.grid_search(["nothing"]), # which parameters are adaptive ? Must be set to nothing, alpha, or tau (which means alpha and tau)
        "mu_DIP" : tune.grid_search([10]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMReg
        "tau_DIP" : tune.grid_search([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMReg. If adaptive tau, it corresponds to tau max
        ## network hyperparameters
        "lr" : tune.grid_search([0.01]), # Learning rate in network optimization
        "sub_iter_DIP" : tune.grid_search([30,100,300]), # Number of epochs in network optimization
        "sub_iter_DIP" : tune.grid_search([100]), # Number of epochs in network optimization
        "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
        "skip_connections" : tune.grid_search([2]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
        "scaling" : tune.grid_search(['standardization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
        "input" : tune.grid_search(['random']), # Neural network input (random or anatomical )
        #"input" : tune.grid_search(["anatomical",'random']), # Neural network input (random or anatomical )
        "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
        "k_DD" : tune.grid_search([32]), # k for Deep Decoder
        ## ADMMReg - OPTITR hyperparameters
        "nb_inner_iteration": tune.grid_search([30]), # Number of inner iterations in DNA and DIPRecon (respectively number of iterations of ADMM-Reg and OPTITR)
        #"nb_inner_iteration": tune.grid_search([3]), # Number of inner iterations in DNA and DIPRecon (respectively number of iterations of ADMM-Reg and OPTITR)
        "nb_inner_iteration": tune.grid_search([10]), # Number of inner iterations in DNA and DIPRecon (respectively number of iterations of ADMM-Reg and OPTITR)
        "alpha" : tune.grid_search([1]), # alpha (penalty parameter) in ADMMReg
        "adaptive_parameters" : tune.grid_search(["both"]), # which parameters are adaptive ? Must be set to nothing, alpha, or both (which means alpha and tau)
        "mu_adaptive" : tune.grid_search([2]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMReg
        "tau" : tune.grid_search([100]), # Factor to multiply alpha in adaptive alpha computation in ADMMReg
        "tau_max" : tune.grid_search([100]), # Maximum value for tau in adaptive tau in ADMMReg
        "stoppingCriterionValue" : tune.grid_search([0]), # Value of the stopping criterion in ADMMReg
        "saveSinogramsUAndV" : tune.grid_search([0]), # 1 means save sinograms u and v from CASToR, otherwise it means do not save them. If adaptive tau, it corresponds to tau max
        ## hyperparameters from CASToR algorithms 
        # Optimization transfer (OPTITR) hyperparameters
        "mlem_sequence" : tune.grid_search([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
        # AML/APPGML hyperparameters
        "A_AML" : tune.grid_search([-100,-500,-10000]), # AML lower bound A
        "A_AML" : tune.grid_search([-10,-100]), # AML lower bound A
        # Post smoothing by CASToR after reconstruction
        "post_smoothing" : tune.grid_search([0]), # Post smoothing by CASToR after reconstruction
        #"post_smoothing" : tune.grid_search([6,9,12,15]), # Post smoothing by CASToR after reconstruction
        # NNEPPS post processing
        "NNEPPS" : tune.grid_search([False]), # NNEPPS post-processing. True or False
    }

    # Dictionary containing list of hyperparameters keys from fixed_config and hyperparameters_config
    split_config = {
        "fixed_hyperparameters" : list(fixed_config.keys()),
        "hyperparameters" : list(hyperparameters_config.keys())
    }
    
    # Merge 3 dictionaries
    config = {**settings_config, **fixed_config, **hyperparameters_config, **split_config}

    return config

config_MIC = config_func_MIC()