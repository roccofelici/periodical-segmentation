# TODO: check dimension of the images
import os
import json
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from PIL import Image
import csv

def get_total_area(name):
    """Return the area (width*height) of the first image found for `name`.

    The function will look for an image file inside
    data/corpora/corpus/{name} if that path is a directory. To avoid
    PIL raising a DecompressionBombError for very large images the
    MAX_IMAGE_PIXELS limit is temporarily disabled for the open
    operation and then restored.
    """
    # build expected path for the named corpus entry
    dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data/corpora/corpus/{name}')

    # if a directory is provided, find a likely image file inside
    if os.path.isdir(dir_path):
        file_path = None
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')):
                file_path = os.path.join(dir_path, fname)
                break
        if file_path is None:
            raise FileNotFoundError(f"No image files found in {dir_path}")
    else:
        # assume it's a file path
        file_path = dir_path

    # temporarily disable PIL decompression bomb check for this open
    old_limit = getattr(Image, 'MAX_IMAGE_PIXELS', None)
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(file_path) as img:
            width, height = img.size
    except Image.DecompressionBombError:
        # As a fallback, try again with the check disabled explicitly
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(file_path) as img:
            width, height = img.size
    finally:
        # restore previous value (could be None or an int)
        Image.MAX_IMAGE_PIXELS = old_limit

    return width * height

def count_images(image_data):
    count = 0
    for id, elements in image_data.items():
        if isinstance(elements, dict) and elements.get('class') == 'image':
            count += 1
    return count

def percentage_image_area(name, image_data):
    sum_areas = 0 
    for id, elements in image_data.items():
        if isinstance(elements, dict) and elements.get('class') == 'image':
            sum_areas += elements.get('area')
    return round(sum_areas/get_total_area(name),3)

# load results
json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/results.json')
if os.path.exists(json_path):
    with open(json_path, 'r') as json_file:
        box_data = json.load(json_file)
else:
    box_data = None

x = [] # number of images
y = [] # percentage of space occupied by images


csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/results_summary.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'year', 'image_count', 'percentage_image_area'])

    if box_data:
        for file_name, indexed_data in box_data.items():
            print(file_name.replace('.jpg',''))
            print(1908) # TO DO: extract from the name
            print(count_images(indexed_data))
            # print(get_total_area(file_name))
            print(percentage_image_area(file_name, indexed_data))

            # - journal name (to be extracted JournalName_Year_PageId)
            # - year (to be extracted NomeGiornale_Anno)
            # - absolute n. of images
            # - percentage of space occupied by images

            writer.writerow([
                file_name, 
                1908, 
                count_images(indexed_data), 
                percentage_image_area(file_name, indexed_data)
                ])



# data viz
source = ColumnDataSource(data=dict(
    x=[1, 2, 3, 4, 5],
    y=[2, 5, 8, 2, 7],
    desc=['A', 'b', 'C', 'd', 'E'],
))

TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("desc", "@desc"),
]

p = figure(width=400, height=400, tooltips=TOOLTIPS,
           title="Mouse over the dots")

p.scatter('x', 'y', size=7, source=source)

# show(p)