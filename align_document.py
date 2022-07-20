from __future__ import print_function
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def orderPoint(pts):
   rect = np.zeros((4,2), dtype = 'float32')

   s = pts.sum(axis=1)
   rect[0] = pts[np.argmin(s)]
   rect[2] = pts[np.argmax(s)]

   diff = np.diff(pts, axis=1)
   rect[1] = pts[np.argmin(diff)]
   rect[3] = pts[np.argmax(diff)]

   return rect

def alignDocument(img):
   src = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
   gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (5,5), 0)
   gray = cv2.Canny(gray, 40, 100)
   (contours, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

   for cnt in cnts:
      epsilon = 0.1 * cv2.arcLength(cnt, True)
      approx = cv2.approxPolyDP(cnt, epsilon, True)
      if len(approx) == 4: # 4:사각형
         cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
         document_detect = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
         break

   rect = orderPoint(approx.reshape(4,2))
   (topLeft, topRight, bottomRight, bottomLeft) = rect

   w1 = abs(bottomRight[0] - bottomLeft[0])
   w2 = abs(topRight[0] - topLeft[0])
   h1 = abs(topRight[1] - bottomRight[1])
   h2 = abs(topLeft[1] - bottomLeft[1])

   maxWidth = int(max([w1, w2]))
   maxHeight = int(max([h1, h2]))

   pst = np.float32([[0,0], [maxWidth-1,0], [maxWidth-1,maxHeight-1], [0,maxHeight-1]])
   matrix = cv2.getPerspectiveTransform(rect, pst)
   dst = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
   dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
   dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

   images = {'Source':src, 'Edge_detection':gray, 'Document_detection':document_detect, 'Result':dst}
   
   return dst, images

if __name__ == "__main__":
   # 명령행 인자 처리
   ap = argparse.ArgumentParser()
   ap.add_argument('-i', '--image', required = True, \
                   help = 'Path to the input image')
   args = vars(ap.parse_args())

   filename = args['image']

   # OpenCV를 사용하여 영상 데이터 로딩
   image = cv2.imread(filename)

   dst, images = alignDocument(image)

   row, col = (2, 2)
   figure = plt.subplots(figsize=(8, 8), constrained_layout=True)
   for i, title in enumerate(images):
      plt.subplot(row, col, i+1)
      plt.title(title)
      plt.imshow(images[title], cmap='gray')
      plt.xticks([]), plt.yticks([])
   plt.show()
