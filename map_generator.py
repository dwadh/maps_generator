import os
import numpy as np
import glob
import scipy.io
from scipy.stats import matrix_normal as mnn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import imageio
from scipy.sparse import csr_matrix
from numpy.linalg import cholesky as chol

class MapGen:
    def __init__(self):
        self.real_maps = []
        self.gen_maps = []
        self.xbar = np.ndarray(shape=(16384,))
        self.var_s = []
        self.subject_dirs = []
        self.data_dir = ""
        self.var_style = {"true_pixel": [],
                    "true_voxel": [],
                    "noise_pixel": [],
                    "noise_voxel": [],
                    "components": [],
                    "voxel_map": []}
        self.var = {}

    def load_data(self, data_dir):
        """
        Load the real maps
        :param
        data_dir: folder with real data
        """

        real_maps = []
        subject_dirs = []
        for subject_dir in os.listdir(data_dir):
            subject_array = np.load(data_dir + "/" + subject_dir + ("/" + subject_dir + "." + "averageMap.pkl"), allow_pickle=True)
            real_maps.append(subject_array.T)
            subject_dirs.append(subject_dir)

        self.real_maps = np.array(real_maps)
        self.subject_dirs = subject_dirs
        self.data_dir = data_dir

    def calculate_global_params(self):
        """
        Calculate the global parameters (xbar, subject variances)
        :return: None
        """

        # Mean map
        self.xbar = np.mean(np.nanmean(self.real_maps, axis=1), axis=0)
        self.real_maps = self.real_maps - self.xbar

        # Variance array
        subject_var = []
        for subject in self.real_maps:
            subject_var.append(np.nanmean((np.nanmean(subject, axis=0) -
                                           self.xbar) ** 2))
        self.var_s = np.array(subject_var)


    def calculate_map_params(self, subject, subject_dir, sub_index):
        """
        Calculate the parameters based on the passed subject
        for generating a new map
        :param
        subject: numpy array containing the pixel values for the five fingers
        subject_dir: folder name containing the subject's noise data

        :return
        subject_component: subject specific component
        finger_component: array with the finger components
        noise_component: noise component for each finger
        noise_pixel: noise components in the pixel space
        data_dict: dictionary consisting of the above listed components
        """

        # Load subject specific matrices
        matrix_dir = self.data_dir + "/" + subject_dir + "/"
        mappingFile = matrix_dir + subject_dir + "." + "voxelMappingInfo.pkl"
        with open(mappingFile, 'rb') as mfile:
          mappingFile = pkl.load(mfile)
        vox2pix = np.array(csr_matrix(mappingFile['vox2Pixel']).todense())
        noise_mat = mappingFile['noise_corr']
        avg_Bvar_est = mappingFile['avg_Bvar_est']
        del mappingFile

        # Generate Subject specific component
        print ("Subject")
        subject_component = np.mean(subject, axis = 0)
        subject_covariance_matrix = np.dot((subject_component.reshape(-1, 1) -
                                            subject_component.mean()),
                                           (subject_component.reshape(1, -1) -
                                            subject_component.mean())) + np.eye(16384) * 0.0000001
        
        # Use cholesky decomposition floowed by matrix multiplication to
        # generate the component with the required correlation structure

        a = chol(subject_covariance_matrix)
        z = np.random.normal(0, 1, size = (16384,))
        z = z/np.std(z)
        subject_component = np.dot(a, z)
        subject_component = (subject_component/subject_component.std()) * (np.sqrt(self.var_s[sub_index]))

        
        # Generate finger specific components
        print ("Finger")
        subject_covariance_matrix = np.dot((subject_component.reshape(-1, 1) -
                                            subject_component.mean()),
                                           (subject_component.reshape(1, -1) -
                                            subject_component.mean())) + np.eye(16384) * 0.0000001
        finger_covariance_matrix = \
            np.ma.cov(np.ma.masked_invalid(subject - subject.mean(axis=0))) + 0.0000001
        finger_component = mnn.rvs(rowcov=finger_covariance_matrix,
                                   colcov=subject_covariance_matrix)
        
        # Generate noise
        print ("Noise")
        noise_list = []
        pixel_noise_list = []
        try:
            for _iter in range(0, 5):
                noise = mnn.rvs(rowcov=noise_mat*avg_Bvar_est)
                noise = noise - noise.mean()
                noise_list.append(noise)
                pixel_noise = np.dot(vox2pix.T, noise)
                pixel_noise_list.append(pixel_noise)
        except Exception as e:
            print (e)
            return -1
        data_dict = {
            "subject_component": subject_component,
            "finger_component": finger_component,
            "noise_component": noise_list,
            "noise_pixel": pixel_noise_list
        }
        del noise_list, finger_component, pixel_noise_list, noise_mat, subject_covariance_matrix
        return data_dict

    def make_maps(self, save_dir):
        """
        Function to generate maps
        :param save_dir: folder to save the data in
        :return: None
        """
        print ("Make Maps")
        generated_maps = []
        for i in range(0, self.real_maps.shape[0]):
            maps_list = []
            data_dict = self.calculate_map_params(self.real_maps[i],
                                                  self.subject_dirs[i], i)
            if data_dict == -1:
                continue

            # Load pixel and voxel transformation matrices

            matrix_dir = self.data_dir + "/" + self.subject_dirs[i] + "/"
            mappingFile = matrix_dir + self.subject_dirs[i] + "." + "voxelMappingInfo.pkl"
            with open(mappingFile, 'rb') as mfile:
              mappingFile = pkl.load(mfile)

            vox2pix = np.array(csr_matrix(mappingFile['vox2Pixel']).todense())
            pix2vox = np.array(csr_matrix(mappingFile['pixel2Vox']).todense())
            del mappingFile

            # Combine xbar and the subject component

            subject_map = self.xbar.reshape(128, 128).T + data_dict['subject_component'].reshape(128, 128)
            true_maps = []

            # Iterate over the finger components and generate the map for each finger

            for finger in range(0, 5):
                pixel_map = subject_map + data_dict['finger_component'][finger].reshape(128, 128)
                voxel_map = np.dot(pix2vox.T, pixel_map.flatten().reshape(-1, 1))
                true_map = pixel_map
                true_maps.append(true_map.flatten())
                voxel_map = voxel_map + data_dict['noise_component'][finger]
                new_map = np.dot(vox2pix.T, voxel_map.flatten())
                maps_list.append(new_map.flatten())

                # Store the component and map variances in the dictionary

                self.var[self.subject_dirs[i]] = self.var_style
                self.var[self.subject_dirs[i]]["components"].append([self.xbar.var(), data_dict['subject_component'].var(),
                                               data_dict['finger_component'][finger].var()])
                self.var[self.subject_dirs[i]]["true_pixel"].append(pixel_map.var())
                self.var[self.subject_dirs[i]]["true_voxel"].append(voxel_map.var())
                self.var[self.subject_dirs[i]]["noise_voxel"].append(data_dict['noise_component'][finger].var())
                self.var[self.subject_dirs[i]]["voxel_map"].append(voxel_map.var())
                
            data_dict['maps'] = maps_list
            data_dict['true_maps'] = true_maps

            # Save the generated maps
            print ("Save")
            self.save_map(data_dict, self.subject_dirs[i], save_dir)
            generated_maps.append(maps_list)

        self.gen_maps = np.array(generated_maps)

    def save_map(self, data_dict, subject_index, save_dir):
        """
        Save the maps to the specified folder as image, matrix
        :param data_dict: dictionary containing the maps and components
        :param subject_index: index of the map set
        :param save_dir: directory to save the maps

        :return: None
        """
        subject_save_dir = save_dir + str(subject_index) + "/"
        subject_maps_dir = subject_save_dir + "maps/"
        if not os.path.exists(subject_maps_dir):
            os.makedirs(subject_maps_dir)
        matrix_loc = subject_maps_dir + "generatedMaps.pkl"
        with open(matrix_loc, mode = "wb") as f:
            pkl.dump(data_dict['maps'], f)

        matrix_loc = subject_maps_dir + "trueMaps.pkl"
        with open(matrix_loc, mode = "wb") as f:
            pkl.dump(data_dict['true_maps'], f)

        subject_maps_dir1 = subject_save_dir + "maps_imgs/"
        if not os.path.exists(subject_maps_dir1):
            os.makedirs(subject_maps_dir1)
        for i in range(0, 5):
            imageio.imsave((subject_maps_dir1 + str(i) + ".png"), data_dict['maps'][i].reshape(128, 128).T)
            print((subject_maps_dir + str(i) + ".png"))

        subject_maps_dir2 = subject_save_dir + "maps_true/"
        if not os.path.exists(subject_maps_dir2):
            os.makedirs(subject_maps_dir2)
        for i in range(0, 5):
            imageio.imsave((subject_maps_dir2 + str(i) + ".png"), data_dict['true_maps'][i].reshape(128, 128).T)
            print((subject_maps_dir + str(i) + ".png"))
        print("Done: ", matrix_loc)


def visualize(data, threshold):
    """
    Scale the data for visualization
    :param data: the array of data to be displayed together
    :param threshold: activation value to set as the neutral/middle color value
    :return: rescaled data for visualizing, on a scale of 0-255 (uint8)
    """
    map_list = []

    # Define the scaler for scaling the value above and below the threshold
    min_scale = MinMaxScaler(feature_range=(0, 127))
    max_scale = MinMaxScaler(feature_range=(128, 255))

    # Fit the data to the scalers
    min_scale = min_scale.fit(data.flatten()[np.where(data.flatten() < threshold)].reshape(-1, 1))
    max_scale = max_scale.fit(data.flatten()[np.where(data.flatten() >= threshold)].reshape(-1, 1))

    # Scale the data
    for i in data:
        i = i.flatten()
        scaled_map = np.ndarray(shape=(i.flatten().shape[0],))
        scaled_map[np.where(i < threshold)] = min_scale.transform(
            i[np.where(i < threshold)].flatten().reshape(-1, 1)).flatten()
        scaled_map[np.where(i >= threshold)] = max_scale.transform(
            i[np.where(i >= threshold)].flatten().reshape(-1, 1)).flatten()
        map_list.append(scaled_map.flatten())
    return (np.uint8(np.array(map_list)))


# Sample code for execution
instance = MapGen()
instance.load_data("../pyData")
instance.calculate_global_params()
instance.make_maps("gen_data_ns_new/")
dict_loc = "gen_data_ns_new_" + "variances.pkl"
with open(dict_loc, mode = "wb") as f:
            pkl.dump(instance.var, f)
# rescaled_maps_for_visualizing = visualize(instance.gen_maps[map_index])