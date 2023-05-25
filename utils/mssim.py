import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from scipy.ndimage import uniform_filter, gaussian_filter


def mssim(img1, img2, alpha, beta, gamma):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    

    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    # window = ndimage.filters.gaussian_filter(np.ones((size, size)), sigma) # Not gaussian filter
    # window = np.ones((size, size))
    window = np.ones((11,11))
    # Normalize uniform filter so that sum is 1
    window /= np.size(window)

    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    L = np.mean(img2) - np.min(img2)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    
    # mu1 = signal.fftconvolve(window, img1, mode='same')
    # mu2 = signal.fftconvolve(window, img2, mode='same')
    mu1 = uniform_filter(img1)
    mu2 = uniform_filter(img2)
        
    
    mu1_sq = mu1*mu1
    mu1_sq = np.multiply(mu1,mu1)
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    # sigma1_sq = signal.fftconvolve(window, img1*img1, mode='same') - mu1_sq
    # sigma2_sq = signal.fftconvolve(window, img2*img2, mode='same') - mu2_sq
    # sigma12 = signal.fftconvolve(window, img1*img2, mode='same') - mu1_mu2
    sigma1_sq = uniform_filter(img1*img1) - mu1_sq
    sigma2_sq = uniform_filter(img2*img2) - mu2_sq
    sigma12 = uniform_filter(img1*img2) - mu1_mu2
    
    # if (np.isnan(np.sum(np.sqrt(sigma1_sq)))):
    #     raise ValueError("NaNs detected in image")
    # if (np.isnan(np.sum(np.sqrt(sigma2_sq)))):
    #     raise ValueError("NaNs detected in image")
    luminance =((2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1))**alpha
    contrast=((2*np.sqrt(sigma1_sq*sigma2_sq) + C2)/(sigma1_sq + sigma2_sq + C2))**beta
    structure=((2*sigma12 + C2)/(2*np.sqrt(sigma1_sq*sigma2_sq) + C2))**gamma
    
    term1=(2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)

    # SSIM=np.mean(luminance*contrast*structure)
    SSIM = np.mean(term1*luminance)
    return SSIM
