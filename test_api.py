import cv2
from gradio_client import Client

client = Client("https://atlury-yolo-document-layout-analysis.hf.space/")


img = cv2.imread("corpora/corpus1/alpha_test.jpg", cv2.IMREAD_COLOR)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


result = client.predict(
				"corpora/corpus1/alpha_test.jpg",	# str (filepath or URL to image) in 'Input Image' Image component
				api_name="/predict"
)
print(result)
