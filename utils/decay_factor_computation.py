import numpy
import math

half_life = 6586.2
lambda_rad = math.log(2) / half_life
duration_acq = 1809

DC = lambda_rad * duration_acq / (1-math.exp(-lambda_rad * duration_acq))

print("decay factor = ", DC)


norm_factor_cylinder = 1.85471e7
ecat7_reco_factor_cylinder = 1.91741e7
norm_factor_brain = 1.85078e7
ecat7_reco_factor_brain = 1.91335e7

print("ratio factor cylinder = ",norm_factor_cylinder / ecat7_reco_factor_cylinder)
print("ratio factor brain = ",norm_factor_brain / ecat7_reco_factor_brain)
print("reversed ratio factor brain = ",ecat7_reco_factor_brain / norm_factor_brain)
