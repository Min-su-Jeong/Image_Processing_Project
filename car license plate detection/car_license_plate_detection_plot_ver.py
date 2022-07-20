# 각 영상의 검출 결과를 시각화: matplotlib의 plot 사용
# python .\car_license_plate_detection_plot_ver.py --dataset dataset --detectset result --detect detect.dat
# python .\accuracy.py --reference ref.dat --detect detect.dat

# 라이브러리 import
import os
import cv2
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

def detectCLP(img):
    approxBefore, approxAfter, res = img.copy(), img.copy(), img.copy()
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 1) 
    canny = cv2.Canny(thresh, 160, 160)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    closed = cv2.dilate(closed, kernel)
    closed = cv2.erode(closed, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    while(True):
        closed = cv2.dilate(closed, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        for c in cnts:
            sx, sy, ex, ey = cv2.boundingRect(c)
            cv2.rectangle(approxBefore, (sx, sy), (sx+ex, sy+ey), (255, 255, 255), 2)

        for c in cnts:
            epsilon = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            sx, sy, ex, ey = cv2.boundingRect(approx)
            cv2.rectangle(approxAfter, (sx, sy), (sx+ex, sy+ey), (255, 255, 255), 2)
            area, ratio = (ex*ey, ex/ey)
            if 16000 < area < 49650 and 1.49 < ratio < 4.9 and 192 < ex < 408 and 63 < ey < 162:
                cv2.rectangle(res, (sx, sy), (sx+ex, sy+ey), (255, 255, 255), 2)
                #---------------------------------------#
                row = 2
                col = 4
                convert = [img, blur, thresh, canny, closed, approxBefore, approxAfter, res]
                title = ['Src', 'Blur', 'Thresh', 'Canny', 'Closed', 'Approx(Before)', 'Approx(After)', 'Result']
                figure = plt.subplots(figsize=(10, 5), constrained_layout=True)
                for i, img in enumerate(convert):
                    plt.subplot(row, col, i+1)
                    plt.title(title[i])
                    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
                    plt.xticks([]), plt.yticks([])
                plt.show()
                #---------------------------------------#
                
                return sx, sy, sx+ex, sy+ey

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True, help = "path to the dataset folder")
    ap.add_argument("-r", "--detectset", required = True, help = "path to the detectset folder")
    ap.add_argument("-f", "--detect", required = True, help = "path to the detect file")
    args = vars(ap.parse_args())
    
    dataset = args["dataset"]
    detectset = args["detectset"]
    detectfile = args["detect"]

    # 결과 영상 저장 폴더 존재 여부 확인
    if(not os.path.isdir(detectset)):
        os.mkdir(detectset)

    # 검출 결과 위치 저장을 위한 파일 생성
    f = open(detectfile, "wt", encoding="UTF-8")  # UTF-8로 인코딩

    # 타이머 시작
    ts = time.time()

    # 번호판 영상에 대한 번호판 영역 검출
    for i, imagePath in enumerate(glob.glob(dataset + "/*.jpg")):
        print(imagePath, '처리중...')

        # 영상을 불러온 후 그레이스케일 영상으로 변환
        ff = np.fromfile(imagePath, np.uint8)
        image = cv2.imdecode(ff, cv2.IMREAD_GRAYSCALE)

        # 번호판 검출
        points = detectCLP(image)

        # 번호판 영역 표시
        detectimg = cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (255, 255, 255), 3)  # 이미지에 사각형 그리기

        # 결과 영상 저장
        loc1 = imagePath.rfind("\\")
        loc2 = imagePath.rfind(".")
        fname = detectset + '/' + str(i) + '_res.jpg'
        cv2.imwrite(fname, detectimg)

        # 검출한 결과 위치 저장
        f.write(imagePath[loc1 + 1: loc2])
        f.write("\t")
        f.write(str(points[0]))
        f.write("\t")
        f.write(str(points[1]))
        f.write("\t")
        f.write(str(points[2]))
        f.write("\t")
        f.write(str(points[3]))
        f.write("\n")
        
    # 영상 처리 소요 시간
    print("Processing time : ", time.time() - ts)
