import numpy as np
import matplotlib.pyplot as plt

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition


if __name__ == "__main__":

    # dataset_path='datasets_two_forces/'
    dataset_path='datasets_rommanager/'

    S = np.load(dataset_path+'FOM.npy')
    print('S shape: '+str(S.shape))

    Phi,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(S.T,1e-16)

    print('Phi shape: '+str(Phi.shape))
    # plt.plot(sigma)
    # plt.semilogy()
    # plt.show()

    print(Phi[:10])

    print()
    

