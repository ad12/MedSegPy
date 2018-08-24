import os
import numpy as np
import cv2
import pickle

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path

def write_ovlp_masks(dir_path, y_true, y_pred):
    dir_path = check_dir(dir_path)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    assert(y_true.shape == y_pred.shape)
    
    num_slices = y_true.shape[0]
    for i in range(num_slices):
        slice_true = y_true[i, :, :]
        slice_pred = y_pred[i, :, :]
        img = generate_ovlp_image(slice_true, slice_pred)
        
        slice_name = '%03d.png' % i
        cv2.imwrite(os.path.join(dir_path, slice_name), img)

def write_mask(dir_path, y_true):
    dir_path = check_dir(dir_path)
    y_true = np.squeeze(y_true) * 255
    num_slices = y_true.shape[0]
    for i in range(num_slices):
        slice_name = '%03d.png' % i
        cv2.imwrite(os.path.join(dir_path, slice_name), y_true[i,:,:])        

def generate_ovlp_image(y_true, y_pred):
    assert(y_true.shape == y_pred.shape)
    assert len(y_true.shape) == 2, "shape should be 2d, but is " + y_true.shape
    
    y_true = y_true.astype(np.bool)
    y_pred = y_pred.astype(np.bool)

    TP = y_true * y_pred
    FN = y_true * (~y_pred)
    FP = (~y_true) * y_pred

    # BGR format
    img = np.stack([FP, TP, FN], axis=-1).astype(np.uint8)
    img *= 255
    return img

def save_pik(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pik(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


