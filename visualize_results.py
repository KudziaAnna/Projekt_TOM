from kits19.starter_code.utils import load_case
import numpy as np
from tqdm import tqdm
import imageio.core.util
import os
import argparse


DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3
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

def visualize_cid_result(case_id,base_dir,specific_img,hu_min=DEFAULT_HU_MIN,hu_max=DEFAULT_HU_MAX,
                         k_color=DEFAULT_KIDNEY_COLOR,t_color=DEFAULT_TUMOR_COLOR,alpha=DEFAULT_OVERLAY_ALPHA):

    path=base_dir + '/'+ str(case_id) + '/'
    check_path_create_if_not_exist(path)

    vol, seg = load_case(case_id)
    spacing = vol.affine
    vol = vol.get_fdata()
    seg = seg.get_fdata()
    seg = seg.astype(np.int32)
    vol_ims = hu_to_grayscale(vol, hu_min, hu_max)
    seg_ims = class_to_color(seg, k_color, t_color)
    viz_ims = overlay(vol_ims, seg_ims, seg, alpha)
    #
    if specific_img != "all":
        case_str="case{}_{}.png".format(case_id,specific_img)
        resultpath = path + case_str
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(viz_ims[int(specific_img)],cmap='gray')
        plt.axis('off')
        plt.title('Provided segmentation')
        #plt.subplot(1,2,2)
        #plt.imshow( cos )
        #plt.axis('off')
        #plt.title('Our segmentation')
        plt.savefig(str(resultpath))
    else:
        for i in tqdm(range(vol.shape[0])):
            case_str = "case{}_{:05d}.png".format(case_id, i)
            resultpath = path + case_str
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(viz_ims[i], cmap='gray')
            plt.axis('off')
            plt.title('Provided segmentation')
            # plt.subplot(1,2,2)
            # plt.imshow( cos )
            # plt.axis('off')
            # plt.title('Our segmentation')
            plt.savefig(str(resultpath))


if __name__ == "__main__":
    desc = "Save image/images that display the effect of our studies"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-c", "--case_id", required=True,
        help="The identifier for the case you would like to visualize"
    )
    parser.add_argument(
        "-d", "--destination", required=False, default = 'results',
        help="The location where you'd like to store the series of pngs"
    )
    parser.add_argument(
        "-s", "--specific_img", required=False, default = 'all',
        help="Specific img you want to compare for given case_id"
    )
    args = parser.parse_args()
    visualize_cid_result(
        args.case_id, args.destination,args.specific_img
    )


