import numpy as np
import kratos_io

# This files calculates the SVD from the snapshot of all the 
# simulations combined to write the proper rom_parameters.json
# which MainKratos_rom.py will use to solve the ROM problem 
# instead of using the SVD from the snapshot of the local
# results


# List of files to read from
data_inputs = [
    "hdf5_bases/result_80000.h5",
    "hdf5_bases/result_90000.h5",
    "hdf5_bases/result_100000.h5",
    "hdf5_bases/result_110000.h5",
    "hdf5_bases/result_120000.h5",
]

# List of variables to run be used (VELOCITY, PRESSURE, etc...)
S, _ = kratos_io.build_snapshot_grid(
    data_inputs, 
    [
        "DISPLACEMENT",
    ]
)

print(f"Readed S with shape: {np.array(S).shape}")

U,_,_ = np.linalg.svd(S, full_matrices=True, compute_uv=True, hermitian=False)
U = U[:,:10]

print(f"Calculated U with {U.shape}")
np.save("svd.npy", U)