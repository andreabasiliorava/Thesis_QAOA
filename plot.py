import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

mypath="n_qubits_4_200step_100_trials"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


file_steps = "qaoa_ising_params_4_level_1_prob_0.78.dat"

A = np.loadtxt(join(mypath, file_steps))

#print(A)
#print(A.shape)
# 
#print(np.where(np.diff(A[:,0]))[0])



split_A = np.split(A, np.where(np.diff(A[:,0]))[0] + 1)

for trial in split_A:
    trial_num = int(trial[0,0])
    energies = trial[:, 4]
    steps = trial[:, 1]
    plt.plot(steps, energies, '.-', label=f"{trial_num:d}")

#plt.legend()
#plt.show()

plt.savefig("plot.pdf", bbox_inches='tight')