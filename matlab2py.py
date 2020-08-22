import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io
from scipy.stats import matrix_normal as mnn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import h5py
from scipy import sparse
import pickle
from scipy.io import savemat

# The following section contains the code to load the the subjects
err_lst = []
done_lst = []
mat_dir = "matlabData1/"
dest_dir = "pyData2/"
subject_list = []
for i in os.listdir(mat_dir):
    name = ""
    subject_list.append(name.join([str(k) + "." for k in i.split(".")[0:3]]))
subject_list = set(subject_list)

# Iterate over all subjects and load the func file for each subject
for i in subject_list:
    subject_save_dir = dest_dir + i[:-1] + "/"

    func_file = mat_dir + str(i) + "func.mat"
    print(func_file)
    try:
        func_file = h5py.File(func_file, 'r')
    except:
        print("\n\nFile not found:", i + "\n\n")
        continue

    # The following section contains the code for extracting the data from the files

    # Calculate average map across runs and extract the voxel noise correlation matrix, avg_Bvar_est
    beta_z = np.asarray(func_file['D']['beta_z'])
    avg_map_voxel = np.mean(beta_z, axis=0)
    noise_cov = np.asarray(func_file['D']['voxNoiseCorr_reg'])
    avg_Bvar_est = func_file['D']['avg_Bvar_est'][0][0]

    # Save the voxel activation values (averaged across runs) to the output pickle file
    voxel_file_name = subject_save_dir + i + "voxelData.pkl"
    if not os.path.exists(subject_save_dir):
        os.makedirs(subject_save_dir)
    with open(voxel_file_name, 'wb') as f:
        pickle.dump(beta_z, f)

    # Load the mapping file
    info_file = mat_dir + str(i) + "voxelMappingInfo.mat"
    print(info_file)
    info_file = h5py.File(info_file, 'r')

    # Load the vox2Pix matrix as a sparse matrix
    vox2Pix = info_file['M']['vox2Pixel']
    vox2Pix = sparse.csc_matrix((vox2Pix['data'], vox2Pix['ir'], vox2Pix['jc']))

    # Check for inconsistencies, if found, load the mapping file from alternate location
    if (vox2Pix.shape[0] != beta_z.shape[1]):
        del vox2Pix
        try:
            print("Load csv")
            vox2Pix = np.loadtxt(("matlabData1/alt_vox/" + str(i) + "vox2Pixel.csv"), delimiter=",")
            print("Loaded csv")
        except:
            print("vox2Pix:", i + "\n\n")
            continue
    del beta_z

    # Load pixel2Vox, node2Pixel, vox2Node matrices as sparse matrices
    pix2Vox = info_file['M']['pixel2Vox']
    pix2Vox = sparse.csc_matrix((pix2Vox['data'], pix2Vox['ir'], pix2Vox['jc']))
    vox2Node = info_file['M']['vox2Node']
    vox2Node = sparse.csc_matrix((vox2Node['data'], vox2Node['ir'], vox2Node['jc']))
    node2Pix = info_file['M']['node2Pixel']

    # Convert the average map from the voxel space to pixel space and save
    node2Pix = sparse.csc_matrix((node2Pix['data'], node2Pix['ir'], node2Pix['jc']))
    avg_map_pixel = vox2Pix.T.dot(avg_map_voxel)
    avg_map_name = subject_save_dir + i + "averageMap.pkl"
    with open(avg_map_name, 'wb') as f:
        pickle.dump(avg_map_pixel, f)
    del avg_map_pixel, avg_map_voxel

    # Generate the dictionary to store the data for the pickle mapping file
    mapping_dict = {
        "vox2Pixel": vox2Pix,
        "pixel2Vox": pix2Vox,
        "vox2Node": vox2Node,
        "node2Pixel": node2Pix,
        "noise_corr": noise_cov,
        "avg_Bvar_est": avg_Bvar_est
    }

    # Save the mapping file using pickle and close the open files
    mapping_file_name = subject_save_dir + i + "voxelMappingInfo.pkl"
    with open(mapping_file_name, 'wb') as f:
        pickle.dump(mapping_dict, f)
    del mapping_dict, vox2Pix, pix2Vox, vox2Node, node2Pix, noise_cov
    func_file.close()
    info_file.close()
    print("Done:", i)
    done_lst.append(i)

# fm, fdf3, ef1
