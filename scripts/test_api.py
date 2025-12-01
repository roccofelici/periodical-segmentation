# import cv2
# from gradio_client import Client

# client = Client("https://atlury-yolo-document-layout-analysis.hf.space/")

# ipath = "data/corpora/corpus/Adolescenza_1.jpg"
# img = cv2.imread(ipath, cv2.IMREAD_COLOR)
# # cv2.imshow("image", img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# result = client.predict(
#     ipath,
#     api_name="/predict"
# )
# print(result)


from gradio_client import Client

client = Client("https://atlury-yolo-document-layout-analysis.hf.space/")
result = client.predict(
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str (filepath or URL to image) in 'Input Image' Image component
				api_name="/predict"
)
print(result)