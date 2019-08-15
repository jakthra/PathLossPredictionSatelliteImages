from utils.drive_test_route_journal import get_training_test_data
import pandas as pd
import numpy as np

import h5py
from datetime import datetime


class fileGen():

    def __init__(self, feature_path, output_path, draw=True):
        self.feature_path = feature_path
        self.output_path = output_path
        self.draw = draw
        self.create_dataset()

    def __load_features(self):
        self.X_df = pd.read_csv(self.feature_path, index_col=0)

    def __load_outputs(self):
        self.Y_df = pd.read_csv(self.output_path, index_col=0)

    def create_dataset(self):
        self.__load_features()
        self.__load_outputs()
        self.__construct_training_test()
        self.__remove_PCI()

        
        self.__standardize()
            
        self.__get_image_index()


    def __construct_training_test(self):
        self.X_df_train, self.Y_df_train, self.X_df_test, self.Y_df_test = get_training_test_data(self.X_df, self.Y_df, draw=self.draw)

    def __remove_PCI(self):
        # Remove PCI column since it's not considered continuous by a discrete label.
        self.X_df = self.X_df.drop(['PCI'], axis=1)
        self.X_df_train = self.X_df_train.drop(['PCI'], axis=1)
        self.X_df_test = self.X_df_test.drop(['PCI'], axis=1)

    def __get_image_index(self):
        # The index are the image names required.
        self.train_image_idx = self.X_df_train.index
        self.test_image_idx = self.X_df_test.index

    def __standardize(self):
        self.std_X = np.std(self.X_df, 0)
        self.mean_X = np.mean(self.X_df, 0)
        self.std_y = np.std(self.Y_df,0)
        self.mean_y = np.mean(self.Y_df,0)

        # No need to standardize discrete labels.
        for key in self.mean_X.index:
            if 'PCI_' in key:
                self.mean_X[key] = 0
                self.std_X[key] = 1

        self.X_df_train = (self.X_df_train - self.mean_X) / self.std_X
        self.X_df_test = (self.X_df_test - self.mean_X) / self.std_X    
        self.Y_df_train = (self.Y_df_train - self.mean_y) / self.std_y
        self.Y_df_test = (self.Y_df_test - self.mean_y) / self.std_y

    def __add_data_to_file(self, training_or_test):
        if training_or_test == 'training':
            X_df = self.X_df_train
            Y_df = self.Y_df_train
            image_idx = self.train_image_idx
            keys = ['training_images', 'training_features', 'training_output']
        elif training_or_test == 'test':
            X_df = self.X_df_test
            Y_df = self.Y_df_test
            image_idx = self.test_image_idx
            keys = ['test_images', 'test_features', 'test_output']
        else:
            raise Exception('Unknown type, must be training or test')
        pointer = 0


        for idx, row in X_df.iterrows():
            #pointer_idx = image_idx[idx]
            try:
                idx_image_path = self.image_path+str(idx)+".png"
                img = img_to_array(load_img(idx_image_path))
                self.f[keys[0]][pointer,:,:,:] = img/255.0
                self.f[keys[1]][pointer,:] = row.values
                self.f[keys[2]][pointer,:] = Y_df.iloc[pointer].values
                pointer += 1
                if pointer % 500 == 0:
                    print('Done {}/{}'.format(pointer,len(X_df)))
            except Exception as e:
                print("idx: {}, pointer: {}".format(idx, pointer))
                raise e
   

        
    def generate_files(self, root_dir='dataset'):
        # Save arrays
        np.save('{}\\training_features.npy'.format(root_dir), self.X_df_train.values)
        np.save('{}\\training_targets.npy'.format(root_dir), self.Y_df_train.values)
        np.save('{}\\test_features.npy'.format(root_dir), self.X_df_test.values)
        np.save('{}\\test_targets.npy'.format(root_dir), self.Y_df_test.values)
        np.save('{}\\features_mu.npy'.format(root_dir), self.mean_X)
        np.save('{}\\features_std.npy'.format(root_dir), self.std_X)
        np.save('{}\\targets_mu.npy'.format(root_dir), self.mean_y)
        np.save('{}\\targets_std.npy'.format(root_dir), self.std_y)
        np.save('{}\\train_image_idx.npy'.format(root_dir), self.train_image_idx)
        np.save('{}\\test_image_idx.npy'.format(root_dir), self.test_image_idx)
        print(self.X_df_train.columns)
        print(self.Y_df_train.columns)
        




def main():    
    FEATURE_PATH = "dataset/feature_matrix_08_08_18.csv"
    OUTPUT_PATH = "dataset/output_matrix_08_08_18.csv"
    file_generator = fileGen(FEATURE_PATH, OUTPUT_PATH, draw=False)
    file_generator.generate_files()




if __name__ == "__main__": main()