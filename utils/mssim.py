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
    sigma = 1.5
    # size = 11
    # window = ndimage.filters.gaussian_filter(np.ones((size, size)), sigma) # Not gaussian filter
    # window = np.ones((size, size))
    # window = np.ones((11,11))
    # Normalize uniform filter so that sum is 1
    # window /= np.size(window)

    K1 = 0.01
    K2 = 0.03
    # L = 255 #bitdepth of image
    L = np.max(img2) - np.min(img2)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    
    # mu1 = signal.fftconvolve(window, img1, mode='same')
    # mu2 = signal.fftconvolve(window, img2, mode='same')
    
    win_size = 7   # backwards compatibility
    mu1 = uniform_filter(img1,size=win_size)
    mu2 = uniform_filter(img2,size=win_size)
    filter_args = {'sigma': sigma, 'truncate': 3.5} # 3.5 is the number of sigmas to match Wang et al. to have filter size=11
    mu1 = gaussian_filter(img1,**filter_args)
    mu2 = gaussian_filter(img2,**filter_args)
        
    
    mu1_sq = mu1*mu1
    # mu1_sq = np.multiply(mu1,mu1)
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    # sigma1_sq = signal.fftconvolve(window, img1*img1, mode='same') - mu1_sq
    # sigma2_sq = signal.fftconvolve(window, img2*img2, mode='same') - mu2_sq
    # sigma12 = signal.fftconvolve(window, img1*img2, mode='same') - mu1_mu2
    sigma1_sq = uniform_filter(img1*img1,size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2*img2,size=win_size) - mu2_sq
    sigma12 = uniform_filter(img1*img2,size=win_size) - mu1_mu2
    sigma1_sq = gaussian_filter(img1*img1,**filter_args) - mu1_sq
    sigma2_sq = gaussian_filter(img2*img2,**filter_args) - mu2_sq
    sigma12 = gaussian_filter(img1*img2,**filter_args) - mu1_mu2
    
    # if (np.isnan(np.sum(np.sqrt(sigma1_sq)))):
    #     raise ValueError("NaNs detected in image")
    # if (np.isnan(np.sum(np.sqrt(sigma2_sq)))):
    #     raise ValueError("NaNs detected in image")
    luminance =((2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1))**alpha
    contrast=((2*np.sqrt(sigma1_sq*sigma2_sq) + C2)/(sigma1_sq + sigma2_sq + C2))**beta
    structure=((2*sigma12 + C2)/(2*np.sqrt(sigma1_sq*sigma2_sq) + C2))**gamma
    
    term1=(2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)

    print("                      ")
    print("ssim_younes")
    print("mu1",np.mean(mu1))

    print("mu1_mu2 jsp",np.mean(uniform_filter(img1*img1)))
    print("mu1_mu2",np.mean(mu1_mu2))
    print("mu1_sq",np.mean(mu1_sq))
    print("mu2_sq",np.mean(mu2_sq))

    print("sigma1_sq",np.min(sigma1_sq))
    print("sigma1_sq",np.mean(sigma1_sq))
    print("sigma1_sq",np.max(sigma1_sq))
    print("sigma2_sq",np.mean(sigma2_sq))
    print("sigma12",np.mean(sigma12))

    print("C1",np.mean(C1))
    print("K1",np.mean(K1))
    print("L",np.mean(L))
    print("A1",np.mean((2*mu1_mu2 + C1)))
    print("B1",np.mean((mu1_sq + mu2_sq + C1)))
    print("luminance",np.mean(luminance))
    print("             ")

    print("(2*sigma12 + C2)",np.min((2*sigma12 + C2)))
    print("(sigma1_sq + sigma2_sq + C2)",np.min((sigma1_sq + sigma2_sq + C2)))
    print("(2*sigma12 + C2)",np.mean((2*sigma12 + C2)))
    print("(sigma1_sq + sigma2_sq + C2)",np.mean((sigma1_sq + sigma2_sq + C2)))
    print("(2*sigma12 + C2)",np.max((2*sigma12 + C2)))
    print("(sigma1_sq + sigma2_sq + C2)",np.max((sigma1_sq + sigma2_sq + C2)))
    print("term1",np.mean(term1))



    # SSIM=np.mean(luminance*contrast*structure)
    SSIM = np.mean(term1*luminance)
    return SSIM
