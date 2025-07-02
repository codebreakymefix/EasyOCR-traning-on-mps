import easyocr
import cv2
from PIL import Image

img = cv2.imread('images/scan.JPG')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
# equalized = cv2.equalizeHist(gray)
# th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=10)

result = img
# pil_img = Image.fromarray(result)
# pil_img.save('images/scanner.png')

reader = easyocr.Reader(['es','en'], gpu=False) # this needs to run only once to load the model into memory

result = reader.readtext(img,detail= 0)
print(result)