import matplotlib.pyplot as plt
import numpy as np
from csv import reader as reader_csv

metrics_file = "VAR_recon_EMV.csv"
metrics_file = "phantom_VAR_recon_EMV_0.1.csv"
metrics_file = "VAR_recon_test_nested_phantom.csv"
nb_it = 250



with open(metrics_file, 'r') as myfile:
    spamreader = reader_csv(myfile,delimiter=',')
    rows_csv = list(spamreader)
    try: # if file is saved as column
        VAR_recon = [float(rows_csv[i][0]) for i in range(0,nb_it)]
    except: # if file is saved as row
        VAR_recon = [float(rows_csv[0][i]) for i in range(0,nb_it)]
        # VAR_recon = np.array(VAR_recon).T

# plt.plot(np.arange(0,900),np.log(VAR_recon))
# plt.plot(np.arange(0,900)[500:],np.log(VAR_recon[500:]))
# plt.plot(np.arange(0,900)[200:400],np.log(VAR_recon[200:400]))
plt.plot(np.arange(0,nb_it),np.log(VAR_recon))
plt.xlabel("iterations")
plt.ylabel("EMV (log scale)")
plt.show()

print("end")