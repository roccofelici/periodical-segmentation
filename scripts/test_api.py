#!/usr/bin/env python3
'''
Usage example:

python scripts/test_api.py data/examples/LetturaSportiva_1912_annoVIII-numeri24-25-26-27-28-pag0_resized.jpg

The call to the API will fail for images of bigger size.

If the script doesn't work visit https://atlury-yolo-document-layout-analysis.hf.space/ and upload manually the image.

If works fine, the output (i.e. the annotated image) will be at: /tmp/gradio/some_folder/annotated_image.png
e.g. /tmp/gradio/e1a769c62de5225695fa5e3304f526c771db0306/tmp592_n3u9.png
'''
import cv2
import sys
from gradio_client import Client

if __name__ == '__main__':

	print('Press key "0" once the image is open to continue')

	if len(sys.argv) > 1:
		PATH = sys.argv[1] # str (filepath or URL to image) in 'Input Image' Image component
	else:
		PATH = 'data/examples/LetturaSportiva_1912_annoVIII-numeri24-25-26-27-28-pag0_resized.jpg'

	client = Client(
		"https://atlury-yolo-document-layout-analysis.hf.space/",
		)

	img = cv2.imread(PATH, cv2.IMREAD_COLOR)
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	result = client.predict(
					PATH, 
					api_name="/predict"
	)
	print(result) # str representing output in 'Output Image' Image component
