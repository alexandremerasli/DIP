import numpy
import math

half_life = 6586.2
lambda_rad = math.log(2) / half_life
duration_acq = 1809

DC = lambda_rad * duration_acq / (1-math.exp(-lambda_rad * duration_acq))

print("decay factor = ", DC)
