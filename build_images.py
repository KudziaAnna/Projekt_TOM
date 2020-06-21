from kits19.starter_code.utils import load_case
from tqdm import tqdm
import imageio.core.util
import matplotlib.pyplot as plt
from utils import *

############ Global settings #############
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 700
DEFAULT_HU_MIN = -700
DEFAULT_IMAGES_FOLDER='images'
TRAIN_SIZE=160
DEV_SIZE=28
TEST_SIZE=22
##########################################

imageio.core.util._precision_warn = silence_imageio_warning


def save_vol_and_seg_from_cid(case_id,base_dir,sub_dir,hu_min=DEFAULT_HU_MIN,hu_max=DEFAULT_HU_MAX):
    if sub_dir=='test':
        volpath = base_dir + "/" + sub_dir + '/case_' + str(case_id) + "/VOL/vol/"
        segpath = base_dir + "/" + sub_dir + '/case_' + str(case_id) + "/SEG/seg/"
    else:
        volpath=base_dir + "/" + sub_dir + '/VOL/vol/'
        segpath = base_dir + "/" + sub_dir + '/SEG/seg/'

    check_path_create_if_not_exist(volpath)
    check_path_create_if_not_exist(segpath)

    vol,seg=load_case(case_id)
    vol = vol.get_fdata()
    vol = standardize_HU(vol,HU_max=hu_max,HU_min=hu_min)
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



        if sub_dir != 'test':
            if case_str in ready_seg:
                print(segpath + case_str + ' already satisfied')
            else:
                plt.imsave(str(segimgpath), seg[i], cmap='gray')

            flip_case_str = "case{}_{:05d}_flipped.png".format(case_id, i)
            flip_volimgpath = volpath + flip_case_str
            flip_segimgpath = segpath + flip_case_str
            flipped_vol = vol[i][:, ::-1]
            flipped_seg = seg[i][:, ::-1]

            if flip_case_str in ready_vol:
                print(volpath + flip_case_str + ' already satisfied')
            else:
                plt.imsave(str(flip_volimgpath), flipped_vol, cmap='gray')

            if flip_case_str in ready_seg:
                print(segpath + flip_case_str + ' already satisfied')
            else:
                plt.imsave(str(flip_segimgpath), flipped_seg, cmap='gray')


if __name__ == "__main__":
    for i in tqdm(range(210)):
        if (i>=0 and i<TRAIN_SIZE):
            folder='train'
        elif (i>=TRAIN_SIZE and i<TRAIN_SIZE+DEV_SIZE):
            folder='dev'
        elif (i>=TRAIN_SIZE+DEV_SIZE and i<210):
            folder='test'
        else:
            raise IndexError('Index out of bound for case number', i)

        save_vol_and_seg_from_cid(i,DEFAULT_IMAGES_FOLDER,folder)
