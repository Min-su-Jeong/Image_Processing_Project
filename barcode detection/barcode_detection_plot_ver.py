# 각 영상의 검출 결과를 시각화: matplotlib의 plot 사용
# pip install imutils
# python .\barcode_detection_plot_ver.py --dataset dataset --detectset result --detect detect.dat
# python .\accuracy.py --reference ref.dat --detect detect.dat

# 라이브러리 import
import numpy as np
import argparse
import imutils
import glob
import cv2
import os
import matplotlib.pyplot as plt
import copy

# Hyperparameter
## 4. blur & threshold parameter
blur1 = 11
blur2 = 11
cth1 = 130
cth2 = 130
th1 = 100
th2 = 255
## 5. kernel parameter
size1 = 57
size2 = 11
## 6. iteration of erode & dilations parameter
iters = 5

def detectBarcode(img):
   ddepth = cv2.cv.CV_64F if imutils.is_cv2() else cv2.CV_64F
   
   # 2. Compute the gradient magnitude representations in both the x and y direction
   gradX = cv2.Sobel(img, ddepth=ddepth, dx=1, dy=0, ksize=-1)
   gradX = cv2.convertScaleAbs(gradX)
   gradY = cv2.Sobel(img, ddepth=ddepth, dx=0, dy=1, ksize=-1)
   gradY = cv2.convertScaleAbs(gradY)
   
   # 3. Subtract the y-gradient from the x-gradient to reveal the barcoded region
   gradientX = cv2.subtract(gradX, gradY)
   gradientY = cv2.subtract(gradY, gradX)

   # 4. Blur and threshold the image
   blurX = cv2.GaussianBlur(gradientX, (blur1, blur2), 0)
   edgedX = cv2.Canny(blurX, cth1, cth2)
   (_, threshX) = cv2.threshold(edgedX, th1, th2, cv2.THRESH_BINARY)
   #blurX = cv2.bilateralFilter(threshX, 10, 50, 50)
   #blurX = cv2.edgePreservingFilter(blurX, flags=1, sigma_s=45, sigma_r=0.2)

   blurY = cv2.GaussianBlur(gradientY, (blur2, blur1), 0)
   edgedY = cv2.Canny(blurY, cth1, cth2)
   (_, threshY) = cv2.threshold(edgedY, th1, th2, cv2.THRESH_BINARY)
   #blurY = cv2.bilateralFilter(threshY, 10, 50, 50)
   #blurY = cv2.edgePreservingFilter(blurY, flags=1, sigma_s=45, sigma_r=0.2)
   #threshY = cv2.adaptiveThreshold(blurY,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)

   # 5. Apply a closing kernel to the thresholded image
   kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (size1, size2))
   closedX = cv2.morphologyEx(threshX, cv2.MORPH_CLOSE, kernelX)
   kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (size2, size1))
   closedY = cv2.morphologyEx(threshY, cv2.MORPH_CLOSE, kernelY)
   
   # 6. Perform a series of dilations and erosions
   closedX = cv2.erode(closedX, kernelX, iterations = iters)
   closedX = cv2.dilate(closedX, kernelX, iterations = iters)
   closedY = cv2.erode(closedY, kernelY, iterations = iters)
   closedY = cv2.dilate(closedY, kernelY, iterations = iters)
   
   # 7. Find the largest contour in the image, which is now presumably the barcode
   contoursX = cv2.findContours(closedX, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   contoursX = imutils.grab_contours(contoursX)
   largestX = np.zeros((4,2), dtype=int) if not contoursX else sorted(contoursX, key = cv2.contourArea, reverse = True)[0]
   contoursY = cv2.findContours(closedY, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   contoursY = imutils.grab_contours(contoursY)
   largestY = np.zeros((4,2), dtype=int) if not contoursY else sorted(contoursY, key = cv2.contourArea, reverse = True)[0]

   # compute the rotated bounding box of the largest contour
   if (len(largestX) > len(largestY)):
      rect = cv2.minAreaRect(largestX)
   else:
      rect = cv2.minAreaRect(largestY)

   box = cv2.boxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
   box = np.int0(box)
   if (box[2][0] > box[0][0]):
      tmp = box[2][0]
      box[2][0] = box[0][0]
      box[0][0] = tmp
   dst = cv2.rectangle(src.copy(), (box[2][0], box[2][1]), (box[0][0], box[0][1]), (0, 255, 0), 2)  # 이미지에 사각형 그리기   
   #---------------------------------------#
   row = 2
   col = 6
   convert = [src, gradX, gradY, gradientX, gradientY, edgedX, edgedY,
              threshX, threshY, closedX, closedY, dst]
   title = ['source', 'gradX', 'gradY', 'gradientX', 'gradientY', 'cannyX', 'CannyY',
              'threshX', 'threshY', 'closedX', 'closedY', 'result']
   figure = plt.subplots(figsize=(10, 6), constrained_layout=True)
   for i, img in enumerate(convert):
      plt.subplot(row, col, i+1)
      plt.title(title[i])
      if i==0 or i==11:
         plt.imshow(img)
      else:
         plt.imshow(img, cmap='gray')
      plt.xticks([]), plt.yticks([])
   plt.show()
   #---------------------------------------#

   return box

if __name__ == '__main__' :
   # 명령행 인자 처리
   ap = argparse.ArgumentParser()
   ap.add_argument("-d", "--dataset", required = True, help = "path to the dataset folder")
   ap.add_argument("-r", "--detectset", required = True, help = "path to the detectset folder")
   ap.add_argument("-f", "--detect", required = True, help = "path to the detect file")
   args = vars(ap.parse_args())

   dataset = args["dataset"]
   detectset = args["detectset"]
   detectfile = args["detect"]

   # 결과 영상 저장 폴더 존재 여부 확인
   if (not os.path.isdir(detectset)):
      os.mkdir(detectset)

   # 결과 영상 표시 여부
   verbose = False

   # 검출 결과 위치 저장을 위한 파일 생성
   f = open(detectfile, "wt", encoding="UTF-8")  # UT-8로 인코딩

   # 바코드 영상에 대한 바코드 영역 검출
   for imagePath in glob.glob(dataset + "/*.jpg"):
      print(imagePath, '처리중...')

      # 영상을 불러오고 그레이 스케일 영상으로 변환
      image = cv2.imread(imagePath)
      # 1. Convert a color image to a grayscale image
      src = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # 바코드 검출
      points = detectBarcode(gray)

      # 바코드 영역 표시
      detectimg = cv2.rectangle(image, (points[2][0], points[2][1]), (points[0][0], points[0][1]), (0, 255, 0), 2)  # 이미지에 사각형 그리기
		
      # 결과 영상 저장
      loc1 = imagePath.rfind("\\")
      loc2 = imagePath.rfind(".")
      fname = detectset + '/' + imagePath[loc1 + 1: loc2] + '_res.jpg'
      cv2.imwrite(fname, detectimg)
      
      # 검출한 결과 위치 저장
      f.write(imagePath[loc1 + 1: loc2])
      f.write("\t")
      f.write(str(points[2][0]))
      f.write("\t")
      f.write(str(points[2][1]))
      f.write("\t")
      f.write(str(points[0][0]))
      f.write("\t")
      f.write(str(points[0][1]))
      f.write("\n")

      if verbose:
         cv2.imshow("image", image)
         cv2.waitKey(0)
