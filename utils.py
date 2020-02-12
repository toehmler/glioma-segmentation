from glob import glob
import SimpleITK as stik
import numpy as np
import json
import os
from sklearn.utils.class_weight import compute_class_weight
import random
from tqdm import tqdm

# TODO add n4 bias correction from og repo


def load_scans(path):
    flair = glob(path + '/*Flair*/*.mha')
    t1 = glob(path + '/*T1.*/*_n4.mha')
    t1c = glob(path + '/*T1c.*/*_n4.mha')
    t2 = glob(path + '/*T2.*/*.mha')
    gt = glob(path + '/*OT*/*.mha')
    paths = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    scans = [stik.GetArrayFromImage(stik.ReadImage(paths[mod])) 
            for mod in range(len(paths))]
    scans = np.array(scans)
    return scans

def norm_scans(scans):
    normed_scans = np.zeros((155, 240, 240, 5)).astype(np.float32)
    normed_scans[:,:,:,4] = scans[4,:,:,:]
    for mod_idx in range(4):
        for slice_idx in range(155):
            normed_slice = norm_slice(scans[mod_idx,slice_idx,:,:])
            normed_scans[slice_idx,:,:,mod_idx] = normed_slice
    return normed_scans

def norm_slice(slice):
    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    img_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(img_nonzero) == 0:
        return slice
    else:
        normed = (slice - np.mean(img_nonzero)) / np.std(img_nonzero)
        return normed

def find_bounds(center, size):
    top = center[0] - ((size - 1) / 2)
    bottom = center[0] + ((size + 1) / 2)
    left = center[1] - ((size - 1) / 2)
    right = center[1] + ((size + 1) / 2)
    bounds = np.array([top, bottom, left, right]).astype(int)
    return bounds

def generate_balanced(scans):
    class_label = 0
    gt = scans[:,:,:,4]
    class_labels = [0, 1, 2, 3, 4]
    arg_labels = [np.argwhere(gt == label) for label in class_labels]
    min_arg = min([len(arg_label) for arg_label in arg_labels])

    patches = []
    labels = []

    for arg_label in arg_labels:
        if len(arg_label) == min_arg:
            for idx in arg_label:
                bounds = find_bounds([idx[1], idx[2]], 65)
                patch = scans[idx[0], bounds[0]:bounds[1], bounds[2]:bounds[3],:4]
                label = int(scans[idx[0], idx[1], idx[2], 4])
                if patch.shape != (65, 65, 4):
                    continue
                if len(np.argwhere(patch == 0)) > (65 * 65):
                    continue
                patches.append(patch)
                labels.append(label)
        else:
            count = 0
            while count < min_arg:
                sample = random.choice(arg_label)
                bounds = find_bounds([sample[1], sample[2]], 65)
                patch = scans[sample[0], bounds[0]:bounds[1], bounds[2]:bounds[3], :4]
                label = int(scans[sample[0], sample[1], sample[2], 4])
                if patch.shape != (65, 65, 4):
                    continue
                if len(np.argwhere(patch == 0)) > (65 * 65):
                    continue
                patches.append(patch)
                labels.append(label)
                count = count + 1
    patches = np.array(patches)
    labels = np.array(labels)
#    print("patches shape: {}".format(patches.shape))
#    print("labels shape: {}".format(labels.shape))
    return patches, labels

def generate_patient_patches(scans, num_patches):
    pbar = tqdm(total=155)
    patches = []
    labels = []
    for patient_slice in scans:
        pbar.update(1)
        gt = patient_slice[:,:,4]
        if len(np.argwhere(gt == 0)) >= 240*240:
            continue
        slice_x, slice_y = training_patches(patient_slice, num_patches)
        patches.extend(slice_x)
        labels.extend(slice_y)
    pbar.close()
    patches = np.array(patches)
    labels = np.array(labels)
        

def rename_pat_dirs(root):
    os.chdir(root)
    pat_dirs = os.listdir(os.getcwd())
    pat_dirs = [name for name in pat_dirs if 'pat' in name.lower()]
    for i, pat_dir in enumerate(pat_dirs):
        os.rename(pat_dir, 'pat{}'.format(i))


if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['root']
    path = glob(root + '/*pat0*')
    scans = load_scans(path[0])
    scans = norm_scans(scans)
    patches = extract_patient_patches(scans)
    print(patches.shape)






