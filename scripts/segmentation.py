#!/usr/bin/env python3
'''
Segment pages for:

- texts
- titles
- images
'''
import torch
import os
import cv2
import sys
import json
import numpy                as np
import matplotlib.pyplot    as plt
import numpy                as np
from ultralytics        import YOLO
from skimage.transform  import resize
from skimage            import img_as_bool
from skimage.morphology import convex_hull_image

SHOW = False

def tableConvexHull(img, masks):
    mask=np.zeros(masks[0].shape,dtype="bool")
    for msk in masks:
        temp=msk.cpu().detach().numpy()
        chull = convex_hull_image(temp)
        mask=np.bitwise_or(mask,chull)
    return mask

def cls_exists(clss, cls):
    indices = torch.where(clss==cls)
    return len(indices[0])>0

def empty_mask(img):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    return np.array(mask, dtype=bool)

def extract_img_mask(img_model, img, config):
    res_dict = {'status' : 1}
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
    res_dict = {'status': 1}
    try:
        for result in model.predict(source=img2, verbose=False, retina_masks=config['rm'], imgsz=config['sz'], conf=config['conf'], stream=True, classes=config['classes']):
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
    response = {'status': 1}
    ans_masks = []
    all_boxes = []
    img2 = img
    
    # paragraph and text masks
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
        all_boxes.append(boxes)
            
    # image and table masks
    res2 = get_predictions(model, img2, configs['imgtab'])
    if res2['status']==-1:
        response['status'] = -1
        return response
    elif res2['status']==0:
        for i in range(2): 
            ans_masks.append(empty_mask(img))
            
    else:
        masks, boxes = res2['masks'], res2['boxes']
        clss = boxes[:, 5]
        all_boxes.append(boxes)
        
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

    # concatenate boxes
    if all_boxes:
        response['boxes'] = torch.cat(all_boxes, dim=0)
    else:
        response['boxes'] = torch.tensor([])
    return response


def overlay(image, mask, color, alpha, resize=None):
    '''
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
    '''
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

def evaluate(
        img_path, 
        model, # =general_model
        img_model, # =image_model
        configs, # =configs
        flags, # =flags
        json_path
        ):
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' could not be loaded. Please check the file path and format.")
    res = get_masks(img, general_model, image_model, flags, configs)

    if res['status']==-1:
        for idx in configs.keys():
            configs[idx]['rm'] = False
        return evaluate(img_path, model, img_model, configs, flags, json_path)
    else:
        masks = res['masks']
        boxes = res['boxes']

    color_map = {
        0 : (255, 0, 0),    # red       - images
        1 : (0, 255, 0),    # green     - titles
        2 : (0, 0, 255),    # blue      - text
        3 : (255, 255, 0),  # yellow    - no idea 
    }

    # for i, mask in enumerate(masks):
    #     img = overlay(image=img, mask=mask, color=color_map[i], alpha=0.4)

    color_class_map = {
        (255, 0, 0) : 'text',    
        (0, 255, 0) : 'title',    
        (0, 0, 255) : 'image',    
    }

    box_results = {}
    for i, box in enumerate(boxes):

        # red - text
        # green - titles
        # blue - images

        # metadata
        box_index   = i
        x1          = int(box[0])
        y1          = int(box[1])
        x2          = int(box[2])
        y2          = int(box[3])
        center      = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        area        = (x2 - x1) * (y2 - y1)
        class_      = int(box[5])

        color = color_map.get(class_, (255, 255, 255))

        # prepare results
        box_results[box_index] = {
            'box': [x1, y1, x2, y2],
            'center': center,
            'area': area,
            'class': color_class_map.get(color)
        }

        overlay_img = img.copy()
        cv2.rectangle(overlay_img, (x1,y1), (x2,y2), color, -1)
        cv2.addWeighted(overlay_img, 0.1, img, 0.9, 0, img)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img, str(i), (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # save in json without overwrite -------------------------------------------
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            try:
                existing_data = json.load(jf)
            except Exception:
                existing_data = {}
    else:
        existing_data = {}

    # Use filename as key, or another unique identifier if needed
    img_key = os.path.basename(img_path)
    existing_data[img_key] = box_results

    with open(json_path, 'w') as jf:
        json.dump(existing_data, jf, indent=4)

    # --------------------------------------------------------------------------


    return img, boxes

def collect_results(output, boxes, filename):

    boxes = boxes.cpu().detach().numpy()
    
    # remember that pixels are counted from the top left corner
    # notice that the unit is pixels for both coordinates and area
    dresults = {}
    dresults['images'] = {}

    global SHOW
    if SHOW: plt.imshow(output)

    # for i, bb in enumerate(boxes):
    #     dresults['images'][str(i)] = {}
    #     dresults['images'][str(i)]['top_left'] = (bb[0].item(), bb[1].item())
    #     dresults['images'][str(i)]['top_right'] = (bb[2].item(), bb[1].item())
    #     dresults['images'][str(i)]['bottom_left'] = (bb[2].item(), bb[3].item())
    #     dresults['images'][str(i)]['bottom_right'] = (bb[0].item(), bb[3].item())
    #     dresults['images'][str(i)]['center'] = (bb[0].item()+(bb[2].item()-bb[0].item())/2, bb[1].item()+(bb[3].item()-bb[1].item())/2)
    #     dresults['images'][str(i)]['area'] = (bb[2].item()-bb[0].item())*(bb[3].item()-bb[1].item())
        
    #     a = "+"
    #     plt.text(bb[0]+(bb[2]-bb[0])/2, bb[1]+(bb[3]-bb[1])/2, str(i), fontsize=20, color='white', weight='bold')
    #     plt.plot(bb[0], bb[1], a, color="white",markersize=5)
    #     plt.plot(bb[0], bb[3], a, color="white",markersize=5)
    #     plt.plot(bb[2], bb[3], a, color="white",markersize=5)
    #     plt.plot(bb[2], bb[1], a, color="white",markersize=5)
    #     plt.plot(bb[0]+(bb[2]-bb[0])/2, bb[1]+(bb[3]-bb[1])/2, "+", color="white",markersize=5)

        # json_object = json.dumps(dresults, indent=4)
        # with open(filename, "w") as outfile:
        #     outfile.write(json_object)
        
    plt.axis('off')
    # plt.savefig(f"results/annotated_images/{filename}")
    if SHOW: plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        PATH = sys.argv[1] # relative path to the folder e.g. data/LetturaSportiva_1912_giu-lug
        NAME = PATH.replace('data/','')
    else:
        # NOTE: change the folder to annotate
        NAME = 'CorriereDeiPiccoli_1908-1909-1910-1913-1916'
    
        # PATH = f'data/final/{NAME}/{NAME}'
        PATH = f'data/final/{NAME}/{NAME}'

    # load results
    # json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/results.json')
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data/{NAME}_segmentation_results.json')

    print(f'json path: {json_path}')

    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            box_data = json.load(json_file)
    else:
        box_data = None
    # ---------------------------------------------------------

    dirname = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dirname, 'models')
    general_model_name = 'e50_aug.pt'
    image_model_name = 'e100_img.pt'

    general_model = YOLO(os.path.join(model_path, general_model_name))
    image_model = YOLO(os.path.join(model_path, image_model_name))

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

    # iterate among pages in the corpus
    # image_path = os.path.join(dirname, f'data/corpora/corpus/')

    image_path = os.path.join(dirname, f'{PATH}')

    for filename in os.listdir(image_path):

        print(f'prep {filename}')
        f = os.path.join(image_path, filename)

        output, boxes = evaluate(img_path=f, model=general_model, img_model=image_model,\
            configs=configs, flags=flags, json_path=json_path)

        collect_results(output, boxes, filename)

        try:
            out_dir = os.path.join(os.path.dirname(image_path), f'{NAME}_annotated')
            os.makedirs(out_dir, exist_ok=True)
            name, ext = os.path.splitext(filename)
            out_name = f"{name}_annotated{ext}"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, output)
            print(f"Saved annotated image to {out_path}")
        except Exception:
            out_path = 'output.png'
            cv2.imwrite(out_path, output)
            print(f"Saved output to {out_path}")
        
        # break

