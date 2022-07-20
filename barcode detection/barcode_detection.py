# pip install imutils
# python .\barcode_detection.py --dataset dataset --detectset result --detect detect.dat
# python .\accuracy.py --reference ref.dat --detect detect.dat

# 라이브러리 import
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import glob
import time
import cv2
import os

############## Hyperparameter ###############
# 4. blur & threshold parameter
blur1 = 11
blur2 = 11
cth1 = 130
cth2 = 130
th1 = 100
th2 = 255
# 5. kernel parameter
size1 = 57
size2 = 11
# 6. iteration of erode & dilations parameter
iters = 5
#############################################

def detectBarcode(img):
   largest = []
   ddepth = cv2.cv.CV_64F if imutils.is_cv2() else cv2.CV_64F
   
   # 2. Compute the gradient magnitude representations in both the x and y direction
   gradX = cv2.Sobel(img, ddepth=ddepth, dx=1, dy=0, ksize=-1)
   gradX = cv2.convertScaleAbs(gradX)
   gradY = cv2.Sobel(img, ddepth=ddepth, dx=0, dy=1, ksize=-1)
   gradY = cv2.convertScaleAbs(gradY)
   
   # 3. Subtract the y-gradient from the x-gradient to reveal the barcoded region
   gradientX = cv2.subtract(gradX, gradY)
   gradientY = cv2.subtract(gradY, gradX)

   # 0: 수평 방향 / 1: 수직 방향
   for i in range(2):
      # 3. Subtract the y-gradient from the x-gradient to reveal the barcoded region
      if i == 0:
         gradient = cv2.subtract(gradX, gradY)
         blur = cv2.GaussianBlur(gradient, (blur1, blur2), 0)
      else:
         gradient = cv2.subtract(gradY, gradX)
         blur = cv2.GaussianBlur(gradient, (blur2, blur1), 0)
         
      # 4. Edge detection and threshold the image   
      edged = cv2.Canny(blur, cth1, cth2)
      (_, thresh) = cv2.threshold(edged, th1, th2, cv2.THRESH_BINARY)  

      # 5. Apply a closing kernel to the thresholded image
      if i == 0:
         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size1, size2))
      else:
         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size2, size1))

      # 6. Perform a series of dilations and erosions
      closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
      closed = cv2.erode(closed, kernel, iterations = iters)
      closed = cv2.dilate(closed, kernel, iterations = iters)
      
      # 7. Find the largest contour in the image, which is now presumably the barcode
      contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      contours = imutils.grab_contours(contours)
      largest.append(np.zeros((4,2), dtype=int) if not contours else sorted(contours, key = cv2.contourArea, reverse = True)[0])

   # compute the rotated bounding box of the largest contour
   if (len(largest[0]) > len(largest[1])):
      rect = cv2.minAreaRect(largest[0])
   else:
      rect = cv2.minAreaRect(largest[1])
      
   box = cv2.boxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
   box = np.int0(box)
   if (box[2][0] > box[0][0]):
      tmp = box[2][0]
      box[2][0] = box[0][0]
      box[0][0] = tmp

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

   # 타이머 시작
   ts = time.time()
   
   # 바코드 영상에 대한 바코드 영역 검출
   for imagePath in glob.glob(dataset + "/*.jpg"):
      print(imagePath, '처리중...')

      # 영상을 불러오고 그레이 스케일 영상으로 변환
      image = cv2.imread(imagePath)
      # 1. Convert a color image to a grayscale image
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

   # 영상 처리 소요 시간
   print("Processing time : ", time.time() - ts)
