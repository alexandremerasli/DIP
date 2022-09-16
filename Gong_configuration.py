from ray import tune

def config_func():
    settings_config2 = {
        "image" : tune.grid_search(['image0']), # Image from database
        "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
        "method" : tune.grid_search(['Gong']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
        "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
        "nb_threads" : tune.grid_search([64]), # Number of desired threads. 0 means all the available threads
        "FLTNB" : tune.grid_search(['double']), # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
        "debug" : False, # Debug mode = run without raytune and with one iteration
        "max_iter" : tune.grid_search([100]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested
        "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
        "finetuning" : tune.grid_search(['last']),
        "experiment" : tune.grid_search([24]),
        "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
        #"f_init" : tune.grid_search(['1_im_value_cropped']),
        "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
        "replicates" : tune.grid_search(list(range(1,1+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
        "average_replicates" : tune.grid_search([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
    }
    # Configuration dictionnary for hyperparameters to tune
    config22 = {
        "rho" : tune.grid_search([0.003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
        ## network hyperparameters
        "lr" : tune.grid_search([1]), # Learning rate in network optimization
        #"lr" : tune.grid_search([0.001,0.041,0.01]), # Learning rate in network optimization
        "sub_iter_DIP" : tune.grid_search([10]), # Number of epochs in network optimization
        "opti_DIP" : tune.grid_search(['LBFGS']), # Optimization algorithm in neural network training (Adam, LBFGS)
        "skip_connections" : tune.grid_search([3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
        "scaling" : tune.grid_search(['normalization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
        "input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
        "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
        "k_DD" : tune.grid_search([32]), # k for Deep Decoder
        ## ADMMLim - OPTITR hyperparameters
        "nb_inner_iteration" : tune.grid_search([2]), # Number of inner iterations in ADMMLim (if mlem_sequence is False) or in OPTITR (for Gong)
        "nb_outer_iteration": tune.grid_search([10]), # Number outer iterations in ADMMLim
        "alpha" : tune.grid_search([0]), # alpha (penalty parameter) in ADMMLim
        ## hyperparameters from CASToR algorithms 
        # Optimization transfer (OPTITR) hyperparameters
        "mlem_sequence" : tune.grid_search([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
        # AML hyperparameters
        "A_AML" : tune.grid_search([-100]), # AML lower bound A
        # Post smoothing by CASToR after reconstruction
        "post_smoothing" : tune.grid_search([0]), # Post smoothing by CASToR after reconstruction
        # NNEPPS post processing
        "NNEPPS" : tune.grid_search([False]), # NNEPPS post-processing. True or False
    }
   
    # Merge 2 dictionaries
    split_config = {
        "hyperparameters" : list(config2.keys())
    }
    config = {**config3, **config2, **split_config}

    return config

def config_func_MIC():

    settings_config2 = {
        "image" : tune.grid_search(['image0']), # Image from database
        "net" : tune.grid_search(['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
        "random_seed" : tune.grid_search([True]), # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
        "method" : tune.grid_search(['Gong']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))
        "processing_unit" : tune.grid_search(['CPU']), # CPU or GPU
        "nb_threads" : tune.grid_search([10]), # Number of desired threads. 0 means all the available threads
        "FLTNB" : tune.grid_search(['double']), # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
        "debug" : False, # Debug mode = run without raytune and with one iteration
        "max_iter" : tune.grid_search([10]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested and Gong
        "nb_subsets" : tune.grid_search([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
        "finetuning" : tune.grid_search(['last']),
        "experiment" : tune.grid_search([24]),
        "image_init_path_without_extension" : tune.grid_search(['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
        #"f_init" : tune.grid_search(['1_im_value_cropped']),
        "penalty" : tune.grid_search(['MRF']), # Penalty used in CASToR for PLL algorithms
        "replicates" : tune.grid_search(list(range(1,1+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
        "average_replicates" : tune.grid_search([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
    }
    # Configuration dictionnary for hyperparameters to tune
    config22 = {
        #"rho" : tune.grid_search([0.0003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
        "rho" : tune.grid_search([0.003,0.0003,0.00003]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
        ## network hyperparameters
        "lr" : tune.grid_search([0.1,0.5]), # Learning rate in network optimization
        "sub_iter_DIP" : tune.grid_search([100]), # Number of epochs in network optimization
        "opti_DIP" : tune.grid_search(['Adam']), # Optimization algorithm in neural network training (Adam, LBFGS)
        #"skip_connections" : tune.grid_search([0]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
        "skip_connections" : tune.grid_search([0,1,2,3]), # Number of skip connections in DIP architecture (0, 1, 2, 3)
        "scaling" : tune.grid_search(['normalization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
        #"input" : tune.grid_search(['random']), # Neural network input (random or CT)
        "input" : tune.grid_search(['CT','random']), # Neural network input (random or CT)
        "d_DD" : tune.grid_search([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
        "k_DD" : tune.grid_search([32]), # k for Deep Decoder
        ## ADMMLim - OPTITR hyperparameters
        "nb_inner_iteration" : tune.grid_search([50]), # Number of inner iterations in ADMMLim (if mlem_sequence is False) or in OPTITR (for Gong)
        "nb_outer_iteration": tune.grid_search([10]), # Number outer iterations in ADMMLim
        "alpha" : tune.grid_search([0.003]), # alpha (penalty parameter) in ADMMLim
        ## hyperparameters from CASToR algorithms 
        # Optimization transfer (OPTITR) hyperparameters
        "mlem_sequence" : tune.grid_search([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
        # AML hyperparameters
        "A_AML" : tune.grid_search([-100]), # AML lower bound A
        # Post smoothing by CASToR after reconstruction
        "post_smoothing" : tune.grid_search([0]), # Post smoothing by CASToR after reconstruction
        # NNEPPS post processing
        "NNEPPS" : tune.grid_search([False]), # NNEPPS post-processing. True or False
    }

    # Merge 2 dictionaries
    split_config = {
        "hyperparameters" : list(config2.keys())
    }
    config = {**config3, **config2, **split_config}

    return config