#!/usr/bin/env python3
'''
TODO: check dimensions of the images found in the page agree with the images 
themselves

TODO: retrieve from the name and the path of the folder of the corpus the url of
the image to upload on flourish

TODO: add tqmd

run as follows:
python scripts/aggregate.py data/LetturaSportiva_1912_giu-lug_segmentation_results.json
'''
import os
import sys
import json
import pandas       as pd
from bokeh.models   import ColumnDataSource
from bokeh.plotting import figure, show
from PIL            import Image

# Increase PIL decompression bomb limit for large images and avoid the following error
# Traceback (most recent call last):
#   File "/home/rocco/dev/dh/periodical-segmentation/scripts/aggregate.py", line 75, in <module>
#     area_images = percentage_image_area(file_name, indexed_data, corpus_name)
#   File "/home/rocco/dev/dh/periodical-segmentation/scripts/aggregate.py", line 39, in percentage_image_area
#     return round(sum_areas/get_total_area(name, corpus_name),3)
#   File "/home/rocco/dev/dh/periodical-segmentation/scripts/aggregate.py", line 24, in get_total_area
#     with Image.open(dir) as img: width, height = img.size
#   File "/home/rocco/dev/dh/periodical-segmentation/.env/lib/python3.10/site-packages/PIL/Image.py", line 3539, in open
#     im = _open_core(fp, filename, prefix, formats)
#   File "/home/rocco/dev/dh/periodical-segmentation/.env/lib/python3.10/site-packages/PIL/Image.py", line 3528, in _open_core
#     _decompression_bomb_check(im.size)
#   File "/home/rocco/dev/dh/periodical-segmentation/.env/lib/python3.10/site-packages/PIL/Image.py", line 3429, in _decompression_bomb_check
#     raise DecompressionBombError(msg)
# PIL.Image.DecompressionBombError: Image size (225767353 pixels) exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.
Image.MAX_IMAGE_PIXELS = None

def get_total_area(name, corpus_name):
    dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data/{corpus_name}/{name}')
    with Image.open(dir) as img: width, height = img.size
    return width*height

def count_images(image_data):
    count = 0
    for id, elements in image_data.items():
        if isinstance(elements, dict) and elements.get('class') == 'image':
            count += 1
    return count

def percentage_image_area(name, image_data, corpus_name):
    sum_areas = 0 
    for id, elements in image_data.items():
        if isinstance(elements, dict) and elements.get('class') == 'image':
            sum_areas += elements.get('area')    
    return round(sum_areas/get_total_area(name, corpus_name),3)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        PATH = sys.argv[1] # relative path to the folder e.g. data/LetturaSportiva_1912_giu-lug_segmentation_results.json
    else:
        PATH = None

    # load results
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), PATH)
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            box_data = json.load(json_file)
    else:
        box_data = None

    corpus_name = PATH.replace('data/','').replace('_segmentation_results.json','')

    df = pd.DataFrame({
        'annotated_image': [],
        'periodic' : [],
        'year' : [],
        'id' : [],
        'n_images': [],         # number of images
        'area_percentage': []   # percentage of space occupied by images
    })
    
    if box_data:
        for file_name, indexed_data in box_data.items():
            
            print(file_name)
        
            unpack = file_name.replace('.jpg','').split('_')
            if len(unpack) == 3:
                periodic, year, id = unpack[0], unpack[1], unpack[2]
            elif len(unpack) > 3:
                print(f"Warning: {file_name} too many values to unpack ***********************************************")
                print(f"saving as {unpack[0], unpack[1], unpack[2]}")
                periodic, year, id = unpack[0], unpack[1], unpack[2]
                input('press any key to go on')
            else:
                print(f"Warning: {file_name} does not have expected format *******************************************")
                input('press any key to go on')
                continue

            # annotated_image_path = # TODO: retrieve from the name and the path of the folder of the corpus
            n_images = count_images(indexed_data)
            area_images = percentage_image_area(file_name, indexed_data, corpus_name)

            df = pd.concat([df, pd.DataFrame({
                'annotated_image': [file_name],
                'periodic': [periodic],
                'year': [year],
                'id': [id],
                'n_images': [n_images],
                'area_percentage': [area_images]
                })], ignore_index=True)

    df.to_csv(f'data/{corpus_name}.csv', index=False)



    