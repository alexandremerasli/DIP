import numpy as np
import pandas as pd
from pathlib import Path
from os.path import isfile
import matplotlib.pyplot as plt

folder_path = "/home/meraslia/workspace_reco/test_dynamic/"
subfolder = "framebyframe64"
subsubfolder = "framebyframe641"

#folder_path = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image2_0/replicate_1/"
#subfolder = "ADMMLim_test_1_frame23"

folder_path = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image2_0/replicate_1/"
subfolder = "ADMMLim_test_1_frame1"

folder_path = "/home/meraslia/workspace_reco/nested_admm/data/Algo/image2_0/replicate_1/"
subfolder = "ADMMLim_test_0.01"

subsubfolder = subfolder

finalOuterIter = 200
alpha_list = np.zeros((finalOuterIter,1))
relativePrimalResidual_list = np.zeros((finalOuterIter,1))
relativeDualResidual_list = np.zeros((finalOuterIter,1))

#'''
for outer_it in range(1,finalOuterIter+1):
    path_adaptive = folder_path + subfolder + "/" + subsubfolder + '_adaptive_it' + format(outer_it) + '.log'
    theLog = pd.read_table(path_adaptive)
    alphaRow = theLog.loc[[0]]
    alphaRowArray = np.array(alphaRow)
    alphaRowString = alphaRowArray[0, 0]
    alpha_list[outer_it - 1] = float(alphaRowString)
    print("alpha",alpha_list[outer_it - 1])
    
fig, ax1 = plt.subplots()
plt.plot(np.arange(1,finalOuterIter+1),np.log10(alpha_list))
plt.title("alpha for " + subfolder)
plt.xlabel("it")
plt.ylabel("alpha (log scale)")
ax1.set_ylim(-13,0)
plt.savefig("alpha for " + subfolder + ".png")
#'''

#'''
for outer_it in range(1,finalOuterIter+1):
    path_adaptive = folder_path + subfolder + "/" + subsubfolder + '_adaptive_it' + format(outer_it) + '.log'
    theLog = pd.read_table(path_adaptive)
    relativePrimalResidualRow = theLog.loc[[4]]
    relativePrimalResidualRowArray = np.array(relativePrimalResidualRow)
    relativePrimalResidualRowString = relativePrimalResidualRowArray[0, 0]
    relativePrimalResidual_list[outer_it - 1] = float(relativePrimalResidualRowString)
    print("relativePrimalResidual",relativePrimalResidual_list[outer_it - 1])

    relativeDualResidualRow = theLog.loc[[6]]
    relativeDualResidualRowArray = np.array(relativeDualResidualRow)
    relativeDualResidualRowString = relativeDualResidualRowArray[0, 0]
    relativeDualResidual_list[outer_it - 1] = float(relativeDualResidualRowString)
    print("relativeDualResidual",relativeDualResidual_list[outer_it - 1])

fig, ax1 = plt.subplots()
plt.plot(np.arange(1,finalOuterIter+1),np.log10(relativePrimalResidual_list))
plt.plot(np.arange(1,finalOuterIter+1),np.log10(relativeDualResidual_list))
plt.legend(["relativePrimalResidual","relativeDualResidual"])
plt.title("relative residuals for " + subfolder)
plt.xlabel("it")
plt.ylabel("relativePrimalResidual (log scale)")
ax1.set_ylim(-2,3)
plt.savefig("relativePrimalResidual for " + subfolder + ".png")
#'''
