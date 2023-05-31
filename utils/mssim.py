import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


def mssim(img1, img2, alpha, beta, gamma):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2
    """
    
    # Convert to float64 to avoid floating point error and negative values in sigma1_sq or sigma2_sq
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Data range
    L = np.max(img2) - np.min(img2)
    
    # Parameters from Wang et al. 2004
    sigma = 1.5
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    
    # Convolve images (gaussian or uniform filter) to get mean for each patch
    # filter_args = {'win_size': 7} # backwards compatibility (value from python implementation)
    # mu1 = uniform_filter(img1,**filter_args)
    # mu2 = uniform_filter(img2,**filter_args)
    filter_args = {'sigma': sigma, 'truncate': 3.5} # 3.5 is the number of sigmas to match Wang et al. to have filter size=11
    mu1 = gaussian_filter(img1,**filter_args)
    mu2 = gaussian_filter(img2,**filter_args)
        
    # Multiply images
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    # Convolve images (gaussian or uniform filter) to get variance and covariance for each patch. Remove negative values coming from floating point errors
    # sigma1_sq = uniform_filter(img1*img1,**filter_args) - mu1_sq
    # sigma2_sq = uniform_filter(img2*img2,**filter_args) - mu2_sq
    # sigma12 = uniform_filter(img1*img2,**filter_args) - mu1_mu2
    sigma1_sq = gaussian_filter(img1*img1,**filter_args) - mu1_sq
    sigma1_sq[sigma1_sq < 0] = 0
    sigma2_sq = gaussian_filter(img2*img2,**filter_args) - mu2_sq
    sigma2_sq[sigma2_sq < 0] = 0
    sigma12 = gaussian_filter(img1*img2,**filter_args) - mu1_mu2
    
    # Compute luminance, contrast and structure for each patch
    luminance =((2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1))**alpha
    contrast=((2*np.sqrt(sigma1_sq*sigma2_sq) + C2)/(sigma1_sq + sigma2_sq + C2))**beta
    structure=((2*sigma12 + C2)/(2*np.sqrt(sigma1_sq*sigma2_sq) + C2))**gamma
    
    # Compute MSSIM
    MSSIM=np.mean(luminance*contrast*structure)
    return MSSIM
