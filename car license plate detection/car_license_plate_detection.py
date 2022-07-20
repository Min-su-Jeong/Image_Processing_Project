# python .\car_license_plate_detection.py --dataset dataset --detectset result --detect detect.dat
# python .\accuracy.py --reference ref.dat --detect detect.dat

# 라이브러리 import
import os
import cv2
import glob
import time
import argparse
import numpy as np

def detectCLP(img):
    # 영상 전처리
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 1) 
    canny = cv2.Canny(thresh, 160, 160)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    closed = cv2.dilate(closed, kernel)
    closed = cv2.erode(closed, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    while True:
        closed = cv2.dilate(closed, kernel)

        # 연결 요소 찾기 및 선택
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        for c in cnts:
            # 모양 근사화
            epsilon = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            sx, sy, ex, ey = cv2.boundingRect(approx)
            area, ratio = (ex*ey, ex/ey)
            
            # 번호판 비율 및 영역의 크기에 맞는 적절한 좌표 값 찾기
            if 16000 < area < 49650 and 1.49 < ratio < 4.9 and 192 < ex < 408 and 63 < ey < 162:
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
        detectimg = cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (255, 255, 255), 3)

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
