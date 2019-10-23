from glob import glob
import SimpleITK as stik
import numpy as np
import json
import os
from sklearn.utils.class_weight import compute_class_weight

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
    # remove extra bg by cropping each volume to size of (146,192,152) 
    return scans[:,1:147, 29:221, 42:194]

def load_test_scans(path):
    flair = glob(path + '/*Flair*/*.mha')
    t1 = glob(path + '/*T1.*/*.mha')
    t1c = glob(path + '/*T1c.*/*.mha')
    t2 = glob(path + '/*T2.*/*.mha')
    gt = glob(path + '/*OT*/*.mha')
    paths = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    scans = [stik.GetArrayFromImage(stik.ReadImage(paths[mod])) 
            for mod in range(len(paths))]
    scans = np.array(scans)
    return scans





    '''
    scans = []
    for i in range(5):
        img = stik.ReadImage(paths[i])
        scans.append(stik.GetArrayFromImage(img))
    '''    
#    return np.array(scans)

def norm_test_scans(scans):
    normed_test_scans = np.zeros((155,240,240,5)).astype(np.float32)
    normed_test_scans[:,:,:,4] = scans[4,:,:,:]
    for mod_idx in range(4):
        for slice_idx in range(155):
            normed_slice = norm_slice(scans[mod_idx,slice_idx,:,:])
            normed_test_scans[slice_idx,:,:,mod_idx] = normed_slice
    return normed_test_scans

def norm_scans(scans):
    normed_scans = np.zeros((146, 192, 152, 5)).astype(np.float32)
    normed_scans[:,:,:,4] = scans[4,:,:,:]
    for mod_idx in range(4):
        for slice_idx in range(146):
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

def training_patches(slice):
    size = 33
    patches = []
    labels = []
    grid = [(h, w) for h in range(192) for w in range(152)]
    for x, y in grid:
        bounds = find_bounds([x, y], size)
        patch = slice[bounds[0]:bounds[1],bounds[2]:bounds[3],:4]
        label = slice[x,y,4]
        if patch.shape == (size,size,4):
            patches.append(patch)
            labels.append(int(label))
    return np.array(patches), np.array(labels)

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
    #rename_pat_dirs(root)
    patient = glob(root + '/*pat0*')
    scans = load_scans(patient[0])


    









