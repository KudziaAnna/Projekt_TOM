from kits19.starter_code.utils import load_case
import numpy as np
from tqdm import tqdm
import imageio.core.util
from imageio import imwrite
import os


DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
TRAIN_SIZE=168
DEV_SIZE=21
TEST_SIZE=21

def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning

def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location
    seg_color[np.equal(segmentation,1)] = k_color
    seg_color[np.equal(segmentation,2)] = t_color
    return seg_color

def overlay(volume_ims, segmentation_ims, segmentation, alpha):
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha*segmentation_ims+(1-alpha)*volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )
    return overlayed

def check_path_create_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
import matplotlib.pyplot as plt
def save_vol_and_seg_from_cid(case_id,base_dir,sub_dir):
    volpath=base_dir + "/" + sub_dir + '/vol/'
    segpath = base_dir + "/" + sub_dir + '/seg/'

    check_path_create_if_not_exist(volpath)
    check_path_create_if_not_exist(segpath)

    vol,seg=load_case(case_id)
    spacing = vol.affine
    vol = vol.get_fdata()
    seg = seg.get_fdata()
    seg = seg.astype(np.int32)

    ready_vol=os.listdir(volpath)
    ready_seg = os.listdir(segpath)

    for i in tqdm(range(vol.shape[0])):
        case_str="case{}_{:05d}.png".format(case_id,i)
        volimgpath = volpath + case_str
        segimgpath = segpath + case_str

        if case_str in ready_vol:
            print(volpath+case_str+' already satisfied')
        else:
            plt.imsave(str(volimgpath), vol[i],cmap='gray')

        if case_str in ready_seg:
            print(segpath+case_str+' already satisfied')
        else:
            plt.imsave(str(segimgpath), seg[i],cmap='gray')

if __name__ == "__main__":
    for i in tqdm(range(210)):
        if (i>=0 and i<TRAIN_SIZE):
            folder='train'
        elif (i>=TRAIN_SIZE and i<TRAIN_SIZE+DEV_SIZE):
            folder='dev'
        elif (i>=TRAIN_SIZE+DEV_SIZE and i<210):
            folder='test'

        save_vol_and_seg_from_cid(i,'images',folder)


