import json
import math
import h5py
import numpy as np

import keras
import tensorflow as tf
from keras import layers
from itertools import repeat

import matplotlib.pyplot as plt

import KratosMultiphysics as KMP
import KratosMultiphysics.gid_output_process as GOP
import KratosMultiphysics.StructuralMechanicsApplication as SMA

no_dw = 0
no_up = 0

def read_snapshot_from_h5py(file_path, variables):
    data = {}

    for variable in variables:
        data[variable] = []

    with h5py.File(file_path, 'r') as f:
        node_indices = f["ModelData"]["Nodes"]["Local"]["Ids"]
        results_dataset = f["ResultsData"]

        number_of_results = len(results_dataset)
        print("Number of results:", number_of_results)
        for i in range(0, number_of_results):

            for variable in variables:
                data[variable].append(results_dataset[str(i)]["NodalSolutionStepData"][variable][()])

    return data

def read_snapshot_from_h5py_as_tensor(file_path, variables):
    data = []
    data_svd = []

    print('VARIABLES', str(variables))

    with h5py.File(file_path, 'r') as f:
        node_indices = f["ModelData"]["Nodes"]["Local"]["Ids"]
        results_dataset = f["ResultsData"]

        number_of_results = len(results_dataset)
        print('Snapshots in individual S:' + str(number_of_results))

        # We only consider part of the samples as first time steps are tipically numerical noise.
        for i in range(no_dw, number_of_results - no_up): # no_dw and no_up are hard-set to 0

            nodal_solution_dataset = results_dataset[str(i)]["NodalSolutionStepData"]
            nodal_size = len(node_indices) # Number of nodes

            row = np.empty(shape=(nodal_size,0))

            for variable in variables:
                if variable == "PRESSURE":
                    this_data = np.array(nodal_solution_dataset[variable]).reshape((nodal_size,1))
                if variable == "VELOCITY":
                    this_data = np.array(nodal_solution_dataset[variable][:,:2])
                if variable == "DISPLACEMENT":
                    this_data = np.array(nodal_solution_dataset[variable][:,:2]) # Matrix of dim: nodal_size x 2. 2 because of componentes X and Y.
                row = np.concatenate((row, this_data), axis=1) # Concatenate horizontally, so that each row contains all variables for a same node

            row = np.reshape(row, (row.shape[0] * row.shape[1])) # Reshape into a single-dimensional array

            data.append(row)
            data_svd.append(row) # data_svd is just a duplicate of data for now

        # In the original there was some attempt at data augmentation by noise. We could implement this separately

    return np.array(data[1:], dtype=np.float64) #, np.array(data_svd[1:], dtype=np.float64) # We discard the first timestep

def read_snapshot_from_h5py_as_pinn_data(file_path, variables):
    ''' For every point in the resuls we create an pair of type:
        (t,p,x,y),(dx,dy)
    '''
    data = []
    sres = []
    time_step = 1
    var = int(file_path.split('/')[-1].split('_')[-1].split('.')[0])/1e5

    with h5py.File(file_path, 'r') as f:
        results_dataset = f["ResultsData"]
        nodal_dataset = f["ModelData"]["Nodes"]["Local"]

        number_of_results = len(results_dataset)
        print(f"Reading {number_of_results} results...")

        for i in range(number_of_results):
            time = i * time_step

            nodal_solution_dataset = results_dataset[str(i)]["NodalSolutionStepData"]
            nodal_size = len(nodal_dataset["Ids"])

            for variable in variables:
                if variable == "DISPLACEMENT":
                    for node in range(len(nodal_dataset["Ids"])):
                        data.append([
                            time,
                            var,
                            nodal_dataset["Coordinates"][node][0],
                            nodal_dataset["Coordinates"][node][1]
                        ])

                        sres.append([
                            nodal_solution_dataset[variable][node][0],
                            nodal_solution_dataset[variable][node][1]
                        ])

    return data, sres

def build_snapshot_grid(result_files, variables):
    data = None

    for snapshot in result_files:
        print("Reading results for",snapshot)
        snapshot_data = read_snapshot_from_h5py_as_tensor(snapshot, variables) # Get S.T matrix for each dataset. Both outputs are copies of S.T
        if data is None:
            data = snapshot_data
        else:
            data = np.concatenate((data,snapshot_data), axis=0)
        print("Ok! ",str(len(data)),"results loaded")

    # Both returns are the same concatenation of original S matrices (discarding 1st element of each)

    return data.T

def build_snapshot_grid_pinn(result_files, variables):
    data = []
    sres = []

    for snapshot in result_files:
        print("Reading results for", snapshot)
        snap_data, snap_sres = read_snapshot_from_h5py_as_pinn_data(snapshot, variables)

        data += snap_data
        sres += snap_sres

        print("Ok! ",str(len(data)),"results loaded")

    return np.array(data).T, np.array(sres).T

def print_npy_snapshot(snapshot_matrix, do_transpose=False):
    with open("snapshot.npy", "wb") as npy_file:
        to_save_data = snapshot_matrix.copy()
        if do_transpose:
            to_save_data = to_save_data.T

        np.save(npy_file, to_save_data)

# Code below here is to generate GiD output files.

def create_out_mdpa(model_part, file_name):
    model_part.AddNodalSolutionStepVariable(KMP.VELOCITY)
    model_part.AddNodalSolutionStepVariable(KMP.PRESSURE)

    model_part.AddNodalSolutionStepVariable(KMP.MESH_VELOCITY)
    model_part.AddNodalSolutionStepVariable(KMP.NEGATIVE_FACE_PRESSURE)

    model_part.AddNodalSolutionStepVariable(KMP.EMBEDDED_VELOCITY)
    model_part.AddNodalSolutionStepVariable(KMP.EXTERNAL_PRESSURE)

    model_part.AddNodalSolutionStepVariable(KMP.MESH_DISPLACEMENT)

    import_flags = KMP.ModelPartIO.READ

    KMP.ModelPartIO(file_name, import_flags).ReadModelPart(model_part)

def print_results_to_gid(model_part, snapshot_matrix, predicted_matrix):

    gid_output = GOP.GiDOutputProcess(
        model_part,
        "PredictDiff",
        KMP.Parameters("""
            {
                "result_file_configuration": {
                    "gidpost_flags": {
                        "GiDPostMode": "GiD_PostAscii",
                        "WriteDeformedMeshFlag": "WriteUndeformed",
                        "WriteConditionsFlag": "WriteConditions",
                        "MultiFileFlag": "SingleFile"
                    },
                    "file_label": "time",
                    "output_control_type": "step",
                    "output_interval": 1.0,
                    "body_output": true,
                    "node_output": false,
                    "skin_output": false,
                    "plane_output": [],
                    "nodal_results": ["PRESSURE", "NEGATIVE_FACE_PRESSURE", "EXTERNAL_PRESSURE", "VELOCITY", "MESH_VELOCITY", "EMBEDDED_VELOCITY", "MESH_DISPLACEMENT"],
                    "nodal_flags_results": ["ISOLATED"],
                    "gauss_point_results": [],
                    "additional_list_files": []
                }
            }
            """
        )
    )

    gid_output.ExecuteInitialize()

    for ts in range(0, snapshot_matrix.shape[0]):
        if True:
            model_part.ProcessInfo[KMP.TIME] = ts
            gid_output.ExecuteBeforeSolutionLoop()
            gid_output.ExecuteInitializeSolutionStep()

            snapshot = snapshot_matrix[ts]
            predicted = predicted_matrix[ts]

            # print("Snapshot size items:", len(snapshot), "-->", len(snapshot) / 3)

            i = 0
            c = 2
            for node in model_part.Nodes:
                node.SetSolutionStepValue(KMP.VELOCITY_X,0,snapshot[i*c+0])
                node.SetSolutionStepValue(KMP.VELOCITY_Y,0,snapshot[i*c+1])
                # node.SetSolutionStepValue(KMP.VELOCITY_Z,0,snapshot[i*c+2])

                node.SetSolutionStepValue(KMP.MESH_VELOCITY_X,0,predicted[i*c+0])
                node.SetSolutionStepValue(KMP.MESH_VELOCITY_Y,0,predicted[i*c+1])
                # node.SetSolutionStepValue(KMP.MESH_VELOCITY_Z,0,predicted[i*c+2])

                node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_X,0,abs(snapshot[i*c+0]-predicted[i*c+0]))
                node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_Y,0,abs(snapshot[i*c+1]-predicted[i*c+1]))
                # node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_Z,0,abs(snapshot[i*c+2]-predicted[i*c+2]))

                i += 1

            gid_output.PrintOutput()
            gid_output.ExecuteFinalizeSolutionStep()

    gid_output.ExecuteFinalize()

def print_results_to_gid_pinn(model_part, sres, pred, num_batchs=5, num_steps=101, print_batch=-1):
    gid_output = GOP.GiDOutputProcess(
        model_part,
        "PredictDiff",
        KMP.Parameters("""
            {
                "result_file_configuration": {
                    "gidpost_flags": {
                        "GiDPostMode": "GiD_PostAscii",
                        "WriteDeformedMeshFlag": "WriteUndeformed",
                        "WriteConditionsFlag": "WriteConditions",
                        "MultiFileFlag": "SingleFile"
                    },
                    "file_label": "time",
                    "output_control_type": "step",
                    "output_interval": 1.0,
                    "body_output": true,
                    "node_output": false,
                    "skin_output": false,
                    "plane_output": [],
                    "nodal_results": ["PRESSURE", "NEGATIVE_FACE_PRESSURE", "EXTERNAL_PRESSURE", "VELOCITY", "MESH_VELOCITY", "EMBEDDED_VELOCITY", "MESH_DISPLACEMENT"],
                    "nodal_flags_results": ["ISOLATED"],
                    "gauss_point_results": [],
                    "additional_list_files": []
                }
            }
            """
        )
    )

    gid_output.ExecuteInitialize()
    model_part.ProcessInfo[KMP.TIME] = 0.0

    for b in range(num_batchs):

        if print_batch == b:
            snapshots_matrix = []

        for s in range(num_steps):
            batch = b * (model_part.NumberOfNodes()*num_steps)
            off   = batch + model_part.NumberOfNodes()*s

            model_part.ProcessInfo[KMP.TIME] += 0.1

            gid_output.ExecuteBeforeSolutionLoop()
            gid_output.ExecuteInitializeSolutionStep()

            if print_batch == b:
                snapshot = []

            if s == 0:
                print("=======================")

            for node in model_part.Nodes:
                nid = node.Id-1

                node.SetSolutionStepValue(KMP.VELOCITY_X,0,sres.T[off+nid][0])
                node.SetSolutionStepValue(KMP.VELOCITY_Y,0,sres.T[off+nid][1])

                node.SetSolutionStepValue(KMP.MESH_VELOCITY_X,0,pred.T[off+nid][0])
                node.SetSolutionStepValue(KMP.MESH_VELOCITY_Y,0,pred.T[off+nid][1])

                node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_X,0,abs(sres.T[off+nid][0]-pred.T[off+nid][0]))
                node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_Y,0,abs(sres.T[off+nid][1]-pred.T[off+nid][1]))

                if s == 0:
                    print(f"P:{node.GetSolutionStepValue(KMP.MESH_VELOCITY_X)}, R:{node.GetSolutionStepValue(KMP.VELOCITY_X)}")

                if print_batch == b:
                    snapshot.append(node.GetSolutionStepValue(KMP.MESH_VELOCITY_X))
                    snapshot.append(node.GetSolutionStepValue(KMP.MESH_VELOCITY_Y))

            if print_batch == b:
                snapshots_matrix.append(snapshot)

            gid_output.PrintOutput()
            gid_output.ExecuteFinalizeSolutionStep()

        if print_batch == b:
            np.save("NN_PINN_ROM.npy", snapshots_matrix)

    gid_output.ExecuteFinalize()