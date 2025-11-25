import torch
import os
import json
import cv2
# import sys
# import wandb
# import requests
# import gradio as gr
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# from tqdm import tqdm
from ultralytics import YOLO
from skimage import img_as_bool
from skimage.transform import resize
from skimage.morphology import convex_hull_image

# wandb.init(mode='disabled')
# np.set_printoptions(threshold=sys.maxsize) # avoid truncation of array display

def tableConvexHull(img, masks):
    mask=np.zeros(masks[0].shape,dtype="bool")
    for msk in masks:
        temp=msk.cpu().detach().numpy();
        chull = convex_hull_image(temp);
        mask=np.bitwise_or(mask,chull)
    return mask

def cls_exists(clss, cls):
    indices = torch.where(clss==cls)
    return len(indices[0])>0

def empty_mask(img):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    return np.array(mask, dtype=bool)

def extract_img_mask(img_model, img, config):
    res_dict = {
        'status' : 1
    }
    res = get_predictions(img_model, img, config)
    
    if res['status']==-1:
        res_dict['status'] = -1
        
    elif res['status']==0:
        res_dict['mask']=empty_mask(img)
        
    else:
        masks = res['masks']
        boxes = res['boxes']
        clss = boxes[:, 5]
        mask = extract_mask(img, masks, boxes, clss, 0)
        res_dict['mask'] = mask
    return res_dict

def get_predictions(model, img2, config):
    res_dict = {
        'status': 1
    }
    try:
        for result in model.predict(source=img2, verbose=False, retina_masks=config['rm'],\
                                    imgsz=config['sz'], conf=config['conf'], stream=True,\
                                    classes=config['classes']):
            try:
                res_dict['masks'] = result.masks.data
                res_dict['boxes'] = result.boxes.data
                del result
                return res_dict
            except Exception as e:
                res_dict['status'] = 0
                return res_dict
    except:
        res_dict['status'] = -1
        return res_dict

def extract_mask(img, masks, boxes, clss, cls):
    if not cls_exists(clss, cls):
        return empty_mask(img)
    indices = torch.where(clss==cls)
    c_masks = masks[indices]
    mask_arr = torch.any(c_masks, dim=0).bool()
    mask_arr = mask_arr.cpu().detach().numpy()
    mask = mask_arr
    return mask

def get_masks(img, model, img_model, flags, configs):
    response = {
        'status': 1
    }
    ans_masks = []
    img2 = img
    
    # getting paragraph and text masks
    res = get_predictions(model, img2, configs['paratext'])

    if res['status']==-1:
        response['status'] = -1
        return response
    elif res['status']==0:
        for i in range(2): ans_masks.append(empty_mask(img))
    else:
        masks, boxes = res['masks'], res['boxes']

        clss = boxes[:, 5]
        for cls in range(2):
            mask = extract_mask(img, masks, boxes, clss, cls)
            ans_masks.append(mask)
                    
    # getting image and table masks
    res2 = get_predictions(model, img2, configs['imgtab'])
    if res2['status']==-1:
        response['status'] = -1
        return response
    elif res2['status']==0:
        for i in range(2): ans_masks.append(empty_mask(img))
    else:
        masks, boxes = res2['masks'], res2['boxes']
        
        # save images bounding boxes
        global bounding_boxes  
        bounding_boxes = boxes 

        clss = boxes[:, 5]
        
        if cls_exists(clss, 2):
            img_res = extract_img_mask(img_model, img, configs['image'])
            if img_res['status'] == 1:
                img_mask = img_res['mask']
            else:
                response['status'] = -1
                return response
            
        else:
            img_mask = empty_mask(img)
        ans_masks.append(img_mask)
        
        if cls_exists(clss, 3):
            indices = torch.where(clss==3)
            tbl_mask = tableConvexHull(img, masks[indices])
        else:
            tbl_mask = empty_mask(img)
        ans_masks.append(tbl_mask)
    
    if not configs['paratext']['rm']:
        h, w, c = img.shape
        for i in range(4):
            ans_masks[i] = img_as_bool(resize(ans_masks[i], (h, w)))
            
    
    response['masks'] = ans_masks
    return response

def overlay(image, mask, color, alpha, resize=None):
    """
    Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray
    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

model_path = 'models'
general_model_name = 'e50_aug.pt'
image_model_name = 'e100_img.pt'

general_model = YOLO(os.path.join(model_path, general_model_name))
image_model = YOLO(os.path.join(model_path, image_model_name))

image_path = 'examples'
sample_name = ['0040da34-25c8-4a5a-a6aa-36733ea3b8eb.png',
               '0050a8ee-382b-447e-9c5b-8506d9507bef.png', 
               '0064d3e2-3ba2-4332-a28f-3a165f2b84b1.png']

sample_path = [os.path.join(image_path, sample) for sample in sample_name]

flags = {
    'hist': False,
    'bz': False
}

configs = {}
configs['paratext'] = {
    'sz' : 640,
    'conf': 0.25,
    'rm': True,
    'classes': [0, 1]
}
configs['imgtab'] = {
    'sz' : 640,
    'conf': 0.35,
    'rm': True,
    'classes': [2, 3]
}
configs['image'] = {
    'sz' : 640,
    'conf': 0.35,
    'rm': True,
    'classes': [0]
}

def evaluate(img_path, model=general_model, img_model=image_model,\
          configs=configs, flags=flags):

    if isinstance(img_path, str):

        img = cv2.imread(img_path) # read image        
        res = get_masks(img, general_model, image_model, flags, configs) # get masks

        if res['status']==-1:
            for idx in configs.keys():
                configs[idx]['rm'] = False
            return evaluate(img, model, img_model, flags, configs)
        else:
            masks = res['masks']
        
        color_map = {
            0 : (255, 0, 0), # R -> images
            1 : (0, 255, 0), # G -> titles, captions (isolated texts)
            2 : (0, 0, 255), # B -> text
            3 : (255, 255, 0), # Y -> ?
        }
        for i, mask in enumerate(masks):
            img = overlay(image=img, mask=mask, color=color_map[i], alpha=0.4)
        # print('finishing')
        return img
    else:
        print(f'we have a problem: {type(img_path), img_path}')
        

def collect_results(path, filename):
    output = evaluate(img_path=path, model=general_model, img_model=image_model,\
          configs=configs, flags=flags)
    
    # remember that pixels are counted from the top left corner
    # notice that the UoM is pixels for both coordinates and area
    dresults = {}
    dresults['images'] = {}

    plt.imshow(output)

    for i, bb in enumerate(bounding_boxes):
        dresults['images'][str(i)] = {}
        dresults['images'][str(i)]['top_left'] = (bb[0].item(), bb[1].item())
        dresults['images'][str(i)]['top_right'] = (bb[2].item(), bb[1].item())
        dresults['images'][str(i)]['bottom_left'] = (bb[2].item(), bb[3].item())
        dresults['images'][str(i)]['bottom_right'] = (bb[0].item(), bb[3].item())
        dresults['images'][str(i)]['center'] = (bb[0].item()+(bb[2].item()-bb[0].item())/2, bb[1].item()+(bb[3].item()-bb[1].item())/2)
        dresults['images'][str(i)]['area'] = (bb[2].item()-bb[0].item())*(bb[3].item()-bb[1].item())
        
        plt.plot(bb[0], bb[1], "+", color="white",markersize=5)
        plt.plot(bb[0], bb[3], "+", color="white",markersize=5)
        plt.plot(bb[2], bb[3], "+", color="white",markersize=5)
        plt.plot(bb[2], bb[1], "+", color="white",markersize=5)
        plt.plot(bb[0]+(bb[2]-bb[0])/2, bb[1]+(bb[3]-bb[1])/2, "+", color="white",markersize=5)

        json_object = json.dumps(dresults, indent=4)
        with open(f"results/data/{filename}.json", "w") as outfile:
            outfile.write(json_object)
        
    plt.axis('off')
    plt.savefig(f"results/annotated_images/{filename}")
    # plt.show()

# iterate among pages in the corpus
directory = 'corpora/corpus'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    collect_results(f, filename)
    break

# # ------------------------------- application -------------------------------
# inputs_image = [
#     gr.components.Image(type="filepath", label="Input Image"),
# ]
# outputs_image = [
#     gr.components.Image(type="numpy", label="Output Image"),
# ]
# interface_image = gr.Interface(
#     fn=evaluate,
#     inputs=inputs_image,
#     outputs=outputs_image,
#     title="MARTA IMAGINATOR MACHINE",
#     examples=sample_path,
#     cache_examples=True,
# ).launch()
