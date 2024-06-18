import glob
import numpy as np
import imageio

def load_images(folder_path):
    files = sorted(glob.glob(folder_path + "/*.png"))
    images = [imageio.imread(file) for file in files]
    return np.array(images)

def split_data(high_res, low_res, test_split=0.1):
    num_samples = len(high_res)
    num_test = int(test_split * num_samples)
    test_idx = np.random.choice(num_samples, num_test, replace=False)
    train_idx = np.setdiff1d(np.arange(num_samples), test_idx)
    
    high_res_train = high_res[train_idx]
    low_res_train = low_res[train_idx]
    high_res_test = high_res[test_idx]
    low_res_test = low_res[test_idx]
    
    return high_res_train, low_res_train, high_res_test, low_res_test
