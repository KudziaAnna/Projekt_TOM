import numpy as np
import os

def standardize_HU(vol, HU_max=512, HU_min=-512):
    vol[vol > HU_max] = HU_max
    vol[vol < HU_min] = HU_min

    conversion_factor = 1.0 / (HU_max - HU_min)
    conversion_intercept = 0.5
    vol = vol * conversion_factor + conversion_intercept

    assert np.amax(vol) <= 1, "Max above one after normalization."
    assert np.amin(vol) >= 0, "Min below zero after normalization."

    return vol

def silence_imageio_warning(*args, **kwargs):
    pass

def check_path_create_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

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