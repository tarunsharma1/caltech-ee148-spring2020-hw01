import os
import numpy as np
import json
from PIL import Image
from PIL import ImageDraw

import matplotlib.pyplot as plt

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    # I extract windows of certain sizes (1x template patch size, 2x template patch size, 4x template patch size)
    # in a sliding window fashion across each image and then take the inner product of each of the resized windows and
    # a template patch. I repeat this for all the template patches I have (3). If there is a red traffic light then the 
    # value of that inner product should be pretty high. I 
    # will use this to find a threshold. The multiple sizes is to make it scale invariant to some extent.


    # normalize
    I = I/255.0

    # lets start with scaling factor = 1
    im_h, im_w,_ = I.shape
    stride = 1
    threshold = 0.9

    print (im_h,im_w)

    # for each manually saved template plate, slide them over images and compare against threshold
    for template_patch in list_of_patches:
        template_patch_h, template_patch_w,_ = template_patch.shape
        print (template_patch_h, template_patch_w)

        scaling_factors = [1,2,4]

        
        template_patch_flattened = np.asarray(template_patch).flatten()
        # we have to divide this by its norm
        template_patch_flattened_norm = np.sqrt(np.sum(template_patch_flattened**2))
        

        for scale in scaling_factors:
            
            for row,i in enumerate(range(0,im_h - template_patch_h, stride)):
                for col,j in enumerate(range(0, im_w - template_patch_w, stride)):

                    window = I[i:i+scale * template_patch_h, j:j+ scale * template_patch_w,:]
                    
                    # resize it such that it is the same size as the template and hence we can compare
                    # i.e in order to get red lights of larger scale, we take larger windows and resize each window
                    # to be the same size as the template in order to compare..this will preserve the aspect ratio
                    img_window = Image.fromarray(np.array(window*255,dtype='uint8')).resize((template_patch_w,template_patch_h))
                    window = np.asarray(img_window)/255.0


                    window_flattened = window.flatten()

                    # we have to divide this by its norm
                    window_flattened_norm = np.sqrt(np.sum(window_flattened**2))

                    inner_product = np.inner(window_flattened, template_patch_flattened)
                    cos_angle = inner_product/(window_flattened_norm*template_patch_flattened_norm)

                    if cos_angle> threshold:
                        bounding_boxes.append([i,j,i+scale*template_patch_h,j+scale*template_patch_w])
            
    







    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    
    # box_height = 8
    # box_width = 6
    
    # num_boxes = np.random.randint(1,5) 
    
    # for i in range(num_boxes):
    #     (n_rows,n_cols,n_channels) = np.shape(I)
        
    #     tl_row = np.random.randint(n_rows - box_height)
    #     tl_col = np.random.randint(n_cols - box_width)
    #     br_row = tl_row + box_height
    #     br_col = tl_col + box_width
        
    #     bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes



def visualize_results(I, bounding_boxes=[]):
    # takes in an image and corresponding bounding boxes, draws the boxes and displays the result
    im = Image.fromarray(I)

    print (bounding_boxes)
    draw = ImageDraw.Draw(im)
    for i in range(0,len(bounding_boxes)):
        pt1,pt2,pt3,pt4 = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], bounding_boxes[i][3]
        #153 316 171 324
        draw.rectangle([(pt2,pt1),(pt4,pt3)], outline="green")
        
    im.show()
    


def find_and_save_patch(I):
    # opens the first image in the list and saves all patchs (traffic lights) in a list
    list_of_patches = []

    width, height = I.size
    print (width, height)
    # normalize
    I = np.asarray(I)
    I = I/255.0
    # I manually found these values
    im1 = I[154:172, 316:324,:]
    im2 = I[192:205,419:427,:]
    im3 = I[179:201,65:81,:]

    list_of_patches.append(im1)
    list_of_patches.append(im2)
    list_of_patches.append(im3)

    return list_of_patches



# set the path to the downloaded data: 
data_path = '/home/tarun/Downloads/caltech-ee148-hw-data/hw1/RedLights2011_Medium'



# set a path for saving predictions: 
preds_path = '/home/tarun/Downloads/caltech-ee148-hw-data/hw1/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# I manually find and crop out the locations of all red lights in the first image to use as templates for the algorithm
list_of_patches = find_and_save_patch(Image.open(os.path.join(data_path,file_names[0])))
#print ('template patch size is ', template_patch.size)
#template_patch_h, template_patch_w,_  = template_patch.shape




preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)

    
    preds[file_names[i]] = detect_red_light(I)
    visualize_results(I, preds[file_names[i]])
    

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
