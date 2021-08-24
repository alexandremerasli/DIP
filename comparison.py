## Python libraries

# Pytorch
import numpy
from torch.utils.tensorboard import SummaryWriter

# Useful
import os

# Math
import numpy as np
import matplotlib.pyplot as plt

# Local files to import
from utils_func import *

## Computing metrics FOR MLEM (must add post smoothing) reconstruction

# castor-recon command line
header_file = ' -df ' + subroot + 'Data/data_eff10/data_eff10.cdh' # PET data path

executable = 'castor-recon'
dim = ' -dim ' + PETImage_shape_str
vox = ' -vox 4,4,4'
vb = ' -vb 3'
it = ' -it 40:6'
th = ' -th 0'
proj = ' -proj incrementalSiddon'
conv = ''

output_path = ' -dout ' + subroot + 'Comparaison/MLEM' # Output path for CASTOR framework
initialimage = ' -img ' + subroot + 'Data/castor_output_it6.hdr'

# Command line for calculating the Likelihood
vb_like = ' -vb 0'
opti_like = ' -opti-fom'

os.system(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage + conv)

# load MLEM previously computed image 
image_MLEM = fijii_np(subroot+'Comparaison/MLEM/MLEM_it40.img', shape=(PETImage_shape))

compute_metrics(image_MLEM,image_gt,i,max_iter,write_tensorboard=False)
write_image_tensorboard(writer,image_MLEM,"Final image computed with MLEM") # MLEM image in tensorboard

# Computing metrics for BSREM reconstruction

# castor-recon command line
header_file = ' -df ' + subroot + 'Data/data_eff10/data_eff10.cdh' # PET data path

executable = 'castor-recon'
dim = ' -dim ' + PETImage_shape_str
vox = ' -vox 4,4,4'
vb = ' -vb 3'
it = ' -it 40:6'
th = ' -th 0'
proj = ' -proj incrementalSiddon'
conv = ''
opti = ' -opti BSREM'
penalty = ' -pnlt MRF'
penaltyStrength = ' -pnlt-beta 1'
ignoreTOF =' ' #'-ignore-TOF'
sauvegardeSens = ''#' -osens'
sauvegardeApresSsEnsemble = '' #-osub'
sauvegardelut = '' #'-olut'

output_path = ' -dout ' + subroot + 'Comparaison/BSREM' # Output path for CASTOR framework
initialimage = ' -img ' + subroot + 'Data/castor_output_it6.hdr'

# Command line for calculating the Likelihood
vb_like = ' -vb 0'
opti_like = ' -opti-fom'

print(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage
          + conv + ignoreTOF + sauvegardeSens + sauvegardeApresSsEnsemble + sauvegardelut + penalty
                      + penaltyStrength)
os.system(executable + dim + vox + output_path + header_file + vb + it + th + proj + opti + opti_like + initialimage
          + conv + ignoreTOF + sauvegardeSens + sauvegardeApresSsEnsemble + sauvegardelut + penalty
                      + penaltyStrength)

# load BSREM previously computed image 
image_BSREM = fijii_np(subroot+'Comparaison/BSREM/BSREM_it40.img', shape=(PETImage_shape))

compute_metrics(image_BSREM,image_gt,i,max_iter,write_tensorboard=False)
write_image_tensorboard(writer,image_BSREM,"Final image computed with BSREM") # BSREM image in tensorboard
