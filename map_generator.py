import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.io
import numpy as np
from scipy.stats import matrix_normal as mnn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MapGen:
    def __init__(self):
        self.real_maps = []
        self.gen_maps = []
        self.surrogates = []
        self.xbar = np.ndarray(shape=(16384,))
        self.var_s = []
        self.subject_dirs = []
        self.data_dir = ""

    def load_data(self, data_dir, surrogates_file):
        """
        Load the real maps and surrogates
        :param 
        data_dir: folder with real data
        surrogates_file: location of surrogates.txt
        """

        real_maps = []
        subject_dirs = []
        for subject_dir in os.listdir(data_dir):
            subject_array = np.ndarray(shape=(5, 16384))
            for map_file in glob.glob(data_dir + "/" + subject_dir + "/*.csv"):
                finger_number = map_file.split("//")[1].split("/")[1]
                finger_index = int(finger_number[1]) - 1
                subject_array[finger_index] = np.loadtxt(map_file,
                                                         delimiter=',').flatten()
            real_maps.append(subject_array)
            subject_dirs.append(subject_dir)

        self.real_maps = np.array(real_maps)
        self.surrogates = np.loadtxt(surrogates_file)
        self.subject_dirs = subject_dirs
        self.data_dir = data_dir

    def calculate_global_params(self):
        """
        Calculate the global parameters (xbar, subject variances)
        :return: None
        """

        # Mean map
        self.xbar = np.mean(np.nanmean(self.real_maps, axis=1), axis=0)

        # Variance matrix
        subject_var = []
        for subject in self.real_maps:
            subject_var.append(np.nanmean((np.nanmean(subject, axis=0) -
                                           self.xbar) ** 2))
        self.var_s = np.array(subject_var)

    def calculate_map_params(self, subject, subject_dir):
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
        noise_visualize: noise maps converted to pixels
        data_dict: dictionary consisting of the above listed components
        """

        # Load subject specific matrices
        matrix_dir = self.data_dir + subject_dir + "/"

        vox2pix_dir = matrix_dir + "vox2Pixel.NN"
        vox2pix = scipy.io.loadmat(vox2pix_dir)
        vox2pix = np.array(vox2pix['vox2Pixel'].todense())

        pix2vox_dir = matrix_dir + "pixel2Vox.NN"
        pix2vox = scipy.io.loadmat(pix2vox_dir)
        pix2vox = np.array(pix2vox['pixel2Vox'].todense())

        noise_dir = matrix_dir + "voxelNoiseCov.NN"
        noise_mat = scipy.io.loadmat(noise_dir)
        noise_mat = noise_mat['S_reg']

        # Generate Subject specific component
        subject_variance = np.nanmean((np.nanmean(subject, axis=0) - self.xbar) ** 2)
        subject_component = self.surrogates[np.random.randint(0, 100)]
        subject_component = subject_component - subject_component.mean()
        subject_component = subject_component * (np.sqrt(subject_variance)
                                                 / np.std(subject_component))

        # Generate finger specific components
        subject_covariance_matrix = np.dot((subject_component.reshape(16384, 1) -
                                            subject_component.mean()),
                                           (subject_component.reshape(1, 16384) -
                                            subject_component.mean())) + np.eye(16384) * 0.0000001
        finger_covariance_matrix = \
            np.ma.cov(np.ma.masked_invalid(subject - subject.mean(axis=0))) + 0.0000001
        finger_component = mnn.rvs(rowcov=finger_covariance_matrix,
                                   colcov=subject_covariance_matrix)

        # Generate noise
        noise_list = []
        pixel_noise_list = []
        try:
            for _iter in range(0, 5):
                noise = mnn.rvs(rowcov=noise_mat)
                noise = noise - noise.mean()
                noise_list.append(noise)
                pixel_noise = np.dot(vox2pix.T, noise)
                pixel_noise_list.append(pixel_noise)
        except:
            return (-1)

        data_dict = {
            "subject_component": subject_component,
            "finger_component": finger_component,
            "noise_component": noise_list,
            "noise_visualize": pixel_noise_list
        }

        return data_dict

    def make_maps(self, save_dir):
        """
        Function to generate maps
        :param save_dir: folder to save the data in
        :return: None
        """

        generated_maps = []
        for i in range(0, self.real_maps.shape[0]):
            maps_list = []
            data_dict = self.calculate_map_params(self.real_maps[i],
                                                  self.subject_dirs[i])
            if (data_dict == -1):
                continue

            # Load pixel and voxel transformation matrices
            matrix_dir = self.data_dir + self.subject_dirs[i] + "/"
            vox2pix_dir = matrix_dir + "vox2Pixel.NN"
            vox2pix = scipy.io.loadmat(vox2pix_dir)
            vox2pix = np.array(vox2pix['vox2Pixel'].todense())

            pix2vox_dir = matrix_dir + "pixel2Vox.NN"
            pix2vox = scipy.io.loadmat(pix2vox_dir)
            pix2vox = np.array(pix2vox['pixel2Vox'].todense())

            # Combine xbar and the subject component
            subject_map = self.xbar.reshape(128, 128) + data_dict['subject_component'].reshape(128, 128)

            # Iterate over the finger components and generated the map for each finger
            for finger in range(0, 5):
                pixel_map = subject_map + data_dict['finger_component'][finger].reshape(128, 128)
                voxel_map = np.dot(pix2vox.T, pixel_map.flatten()) + \
                            data_dict['noise_component'][finger].flatten()
                new_map = np.dot(vox2pix.T, voxel_map.flatten())
                maps_list.append(new_map.flatten())

            data_dict['maps'] = maps_list
            self.save_map(data_dict, i, save_dir)
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
        subject_matrices_dir = subject_save_dir + "matrices/"
        subject_maps_dir = subject_save_dir + "maps/"
        if not os.path.exists(subject_maps_dir):
            os.makedirs(subject_maps_dir)
        if not os.path.exists(subject_matrices_dir):
            os.makedirs(subject_matrices_dir)
        for i in range(0, 5):
            matrix_loc = subject_matrices_dir + str(i) + ".csv"
            pd.DataFrame(data_dict['maps'][i].reshape(128, 128)).to_csv(matrix_loc)
            imageio.imsave((subject_maps_dir + str(i) + ".png"), data_dict['maps'][i].reshape(128, 128))
            print((subject_maps_dir + str(i) + ".png"))
