from ray import tune

def config_func_MIC():

    # Configuration dictionnary for general settings parameters (not hyperparameters)
    settings_config = {
        "image" : tune.grid_search(['image2_0']), # Image from database
        "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
        "method" : tune.grid_search(['Gong']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
        "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
        "nb_threads" : tune.grid_search([1]), # Number of desired threads. 0 means all the available threads
        "FLTNB" : tune.grid_search(['double']), # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
        "debug" : False, # Debug mode = run without raytune and with one iteration
        "ray" : True, # Ray mode = run with raytune if True, to run several settings in parallel
        "tensorboard" : True, # Tensorboard mode = show results in tensorboard
        "all_images_DIP" : tune.grid_search(['Last']), # Option to store only 10 images like in tensorboard (quicker, for visualization, set it to "True" by default). Can be set to "True", "False", "Last" (store only last image)
        "experiment" : tune.grid_search([24]),
        #"f_init" : tune.grid_search(['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
        "f_init" : tune.grid_search(['out_DIP-1_FINAL']),
        "replicates" : tune.grid_search(list(range(1,100+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
        #"replicates" : tune.grid_search(list(range(1,1+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
        "average_replicates" : tune.grid_search([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
        "castor_foms" : tune.grid_search([True]), # Set to True to compute CASToR Figure Of Merits (likelihood, residuals for ADMMLim)
    }
    # Configuration dictionnary for previous hyperparameters, but fixed to simplify
    fixed_config = {
        "max_iter" : tune.grid_search([100]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested and Gong
        "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
        "finetuning" : tune.grid_search(['last']),
        "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
        "unnested_1st_global_iter" : tune.grid_search([False]), # If True, unnested are computed after 1st global iteration (because rho is set to 0). If False, needs to set f_init to initialize the network, as in Gong paper, and rho is not changed.
        "sub_iter_DIP_initial" : tune.grid_search([300]), # Number of epochs in first global iteration (pre iteraiton) in network optimization (only for Gong for now)
        "nb_inner_iteration" : tune.grid_search([1]), # Number of inner iterations in ADMMLim (if mlem_sequence is False). (3 sub iterations are done within 1 inner iteration in CASToR)
        "xi" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in ADMMLim
        "xi_DIP" : tune.grid_search([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in Gong and nested
        "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
        "DIP_early_stopping" : tune.grid_search([False]), # Use DIP early stopping with WMV strategy
        "windowSize" : tune.grid_search([10]), # Network to use (DIP,DD,DD_AE,DIP_VAE)
        "patienceNumber" : tune.grid_search([100]), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    }
    # Configuration dictionnary for hyperparameters to tune
    hyperparameters_config = {
        "image_init_path_without_extension" : tune.grid_search(['BSREM_it30']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
        "rho" : tune.grid_search([0.003,8e-4,0.008,0.03]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
        "rho" : tune.grid_search([0.003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
        "adaptive_parameters_DIP" : tune.grid_search(["nothing"]), # which parameters are adaptive ? Must be set to nothing, alpha, or tau (which means alpha and tau)
        "mu_DIP" : tune.grid_search([10]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMLim
        "tau_DIP" : tune.grid_search([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMLim. If adaptive tau, it corresponds to tau max
        ## network hyperparameters
        "lr" : tune.grid_search([0.01]), # Learning rate in network optimization
        "sub_iter_DIP" : tune.grid_search([100]), # Number of epochs in network optimization
        "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
        "skip_connections" : tune.grid_search([3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
        "scaling" : tune.grid_search(['standardization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
        #"scaling" : tune.grid_search(['standardization','positive_normalization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
        "input" : tune.grid_search(['random']), # Neural network input (random or CT)
        #"input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
        "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
        "k_DD" : tune.grid_search([32]), # k for Deep Decoder
        ## ADMMLim - OPTITR hyperparameters
        "nb_outer_iteration": tune.grid_search([30]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
        #"nb_outer_iteration": tune.grid_search([3]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
        "nb_outer_iteration": tune.grid_search([2]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
        "alpha" : tune.grid_search([1]), # alpha (penalty parameter) in ADMMLim
        "adaptive_parameters" : tune.grid_search(["tau"]), # which parameters are adaptive ? Must be set to nothing, alpha, or tau (which means alpha and tau)
        "mu_adaptive" : tune.grid_search([2]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMLim
        "tau" : tune.grid_search([100]), # Factor to multiply alpha in adaptive alpha computation in ADMMLim. If adaptive tau, it corresponds to tau max
        ## hyperparameters from CASToR algorithms 
        # Optimization transfer (OPTITR) hyperparameters
        "mlem_sequence" : tune.grid_search([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
        # AML/APGMAP hyperparameters
        "A_AML" : tune.grid_search([-100,-500,-10000]), # AML lower bound A
        "A_AML" : tune.grid_search([-10,-100]), # AML lower bound A
        # Post smoothing by CASToR after reconstruction
        "post_smoothing" : tune.grid_search([0]), # Post smoothing by CASToR after reconstruction
        #"post_smoothing" : tune.grid_search([6,9,12,15]), # Post smoothing by CASToR after reconstruction
        # NNEPPS post processing
        "NNEPPS" : tune.grid_search([False]), # NNEPPS post-processing. True or False
    }

    # Merge 3 dictionaries
    split_config = {
        "fixed_hyperparameters" : list(fixed_config.keys()),
        "hyperparameters" : list(hyperparameters_config.keys())
    }
    config = {**settings_config, **fixed_config, **hyperparameters_config, **split_config}

    return config