import numpy as np
import argparse
import os
from glob import glob
import shutil
from PIL import Image
import cv2
import re
#args = None
#parser = argparse.ArgumentParser(description='Mask2Box Tranformation')
#parser.add_argument('-t', '--transform',
#                    type=str, choices=['to_wh'], default=None, help='Wether to return x1y1x2y2 or xywh format')
#parser.add_argument('-d', '--data-directory',
#                    default='../data/OTB/Man', type=str,
#                    help='path to video frames')



def rearrange_files(parent_folder):
    import os
    import shutil

    # list of file paths
    files = sorted(glob(parent_folder+'/*'))
    # create a dictionary to group files by name
    files_dict = {}
    for file in files:
        name = os.path.basename(file).split('-')[0]
        if name in files_dict:
            files_dict[name].append(file)
        else:
            files_dict[name] = [file]

    # create a folder for each group of files and move them into it
    for name, file_list in files_dict.items():
        folder_path = os.path.join(os.path.dirname(file_list[0]), name)
        os.makedirs(folder_path, exist_ok=True)
        for file in file_list:
            shutil.move(file, folder_path)




def get_bounding_box(mask,transform = None):
    """
    Given a binary mask, returns the coordinates of the smallest bounding box
    that contains all the foreground pixels in the mask.

    Args:
        mask: A binary numpy array representing the mask.

    Returns:
        A tuple (x_min, y_min, x_max, y_max) representing the bounding box coordinates.
    """
    assert len(mask.shape) == 2, "Mask must be a 2D binary numpy array."
    assert np.all(np.logical_or(mask == 0, mask == 1)), "Mask must be a binary numpy array."

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    if transform == None : 
        return x_min, y_min, x_max, y_max
    elif transform == "to_wh" :
        return  x_min, y_min, x_max - x_min, y_max - y_min


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: iou

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1_x1,box2_x1)
    yi1 = max(box1_y1,box2_y1)
    xi2 = min(box1_x2,box2_x2)
    yi2 = min(box1_x2,box2_x2)
    inter_width = xi2-xi1
    inter_height = yi2-yi1
    inter_area = max(inter_width, 0) * max(inter_height, 0)
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_y2-box1_y1)*(box1_x2-box1_x1)
    box2_area = (box2_y2-box2_y1)*(box2_x2-box2_x1)
    union_area = box1_area+box2_area-inter_area
    
    # compute the IoU
    iou = inter_area/union_area
    ### END CODE HERE
    
    return iou


if __name__ == "__main__":
    #args = parser.parse_args()
    #print(args)
    print(os.getcwd())
    parent_folder = os.getcwd()+'/sequences-test/sequences-test'
    #rearrange_files(parent_folder)
    for folder in sorted(os.listdir(parent_folder)):
        masks = sorted(glob(parent_folder+'/'+folder+'/*.png'))
        with open(parent_folder+'/'+folder+'.ann','a') as f : 
            for mask_file in masks : 
                print()
                mask = Image.open(mask_file)
                mask = np.int_(np.array(mask)/255)
                print(mask_file,type(mask),mask.shape,np.max(mask),np.min(mask))
                x,y,w,h = get_bounding_box(mask,transform = 'to_wh')
                f.write(f'{x},{y},{w},{h}\n')
        f.close()

    
            


    
