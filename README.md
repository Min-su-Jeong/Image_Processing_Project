# Image_Processing_Project
### Image processing vision project using python OpenCV<br/>
> skin_detector(피부색 영역 검출)<br/>   
> align_document(문서 정렬)<br/>   
> barcode detection(바코드 검출)<br/>  
> car license plate detection(자동차 번호판 검출)<br/> 

<br/> 

## Develop version
- Python: 3.7.13
- Opencv-python: 4.4.0.46
- Numpy: 1.18.5
- Matplotlib: 3.5.2
- Imutils: 0.5.4

<br/>

## Detail Project
### 1. Skin detector(피부색 영역 검출)
- 개요
```
1. 비디오 또는 카메라 영상(컬러 영상)에서 피부색 영역을 검출하여 화면에 출력
2. 주어진 범위에서의 마스크 생성 및 노이즈 제거 작업(가우시안 평활화 사용)을 수행한 후, bitwise and 연산을 수행하여 피부색 영역만을 검출
```  

- 문제 해결 과정
```
1. 파일명: skin_detector.py로 지정 및 필요한 패키지 import
  - argparse, opencv, numpy 등

2. 명령 행 인자를 사용하여 입력 비디오 파일명을 전달한다. 비디오 파일은 옵션으로 생략 가능하며 생략한 경우 카메라를 입력으로 사용한다.
  - ap.add_argument("-v", "--video", required = False, help = "path to the video file")
  - 비디오 파일 옵션 시
    👉 camera = cv2.VideoCapture(args["video"])
  - 비디오 파일 옵션 생략 시
    👉 camera = cv2.VideoCapture(0)

3. 피부색 범위를 설정한 후 히스토그램 슬라이싱을 수행한다.
  - lowerb = np.array([0, 48, 80]) 
  - upperb = np.array([20, 255, 255])
  - dst = histogram_slicing(hsv, lowerb, upperb)
  💡 피부색으로 인식할 최소/최대 범위를 적절히 조절하는 것이 Point!

4. 추출한 피부색 영역에 대한 마스크를 사용하여 원본 영상에서 피부색 영역만을 추출한다.
  - mask = cv2.inRange(img, lowerb, upperb)
  - blur = cv2.GaussianBlur(mask, (5, 5), 0)
  - dst = cv2.bitwise_and(img, img, mask=blur)

5. 비디오 파일의 마지막 또는 카메라를 사용하는 경우 ‘q’키를 누를 때까지 피부색 추출 과정을 반복한다.
  - while True를 사용하여 무한 반복
  - 내장 카메라 사용: (retval, frame) = camera.read()
  - if cv2.waitKey(1) & 0xFF == ord("q"): break 👉 ‘q’ 키를 누르면 종료
```

- 실행 결과
<img align="left" width="320" height="190" src="https://user-images.githubusercontent.com/74342121/180007605-15587a88-f7e0-46b5-bfe6-12c618cb68ec.png">

<img align="left" width="320" height="190" src="https://user-images.githubusercontent.com/74342121/180007836-9ebf5d61-26f6-4b44-8750-08e465d538b4.png">

<img align="center" width="320" height="190" src="https://user-images.githubusercontent.com/74342121/180007849-bc084706-e38e-44d6-bc77-739e250759c8.png">

<br/><br/><hr/>

### 2. Align document(문서 정렬)
- 개요
```
임의 모양 사각형 형태를 갖는 문서의 모양을 직사각형으로 변환한 후 이진화하여 표현
```  

- 문제 해결 과정
```
1. 파일명: align_document.py로 지정 및 필요한 라이브러리 import
  - argparse, opencv, numpy,  matplotlib 등

2. 명령 행 인자를 사용하여 입력 파일명을 전달한다.
  - ap.add_argument('-i', '--image', required = True, help = 'Path to the input image')
  - filename = args[‘image’]

3. 컬러 영상을 그레이 스케일 영상으로 변환한 후 영상 평활화를 수행한다.
  - cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
  - cv2.GaussianBlur(gray, (5,5), 0)

4. 에지 검출을 수행한다.
  - cv2.Canny(gray, 40, 100)

5. 크기가 큰 몇 개의 연결 요소를 선택한 후 모양을 근사화한다.
  - (contours, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 👉 연결 요소 찾기
  - cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5] 👉 연결 요소들 중에서 크기가 가장 큰 5개의 연결 요소 선택
  - approx = cv2.approxPolyDP(cnt, epsilon, True) 👉 모양 근사화

6. 네 개의 꼭짓점을 갖는 크기가 큰 연결 요소를 선택한다.
  - if len(approx) == 4: 👉 네 개의 꼭짓점을 갖는 경우
      cv2.drawContours(img, [approx], 0, (0, 255, 0), 2) 👉 bounding box 그리기

7. 앞에서 선택한 네 개의 꼭짓점과 결과 영상의 네 개 꼭짓점을 대응점으로 사용하여 투영 변환을 수행한다.
  - rect = orderPoint(approx.reshape(4,2))
  - (topLeft, topRight, bottomRight, bottomLeft) = rect
                             ⋮
  - matrix = cv2.getPerspectiveTransform(rect, pst)

8. 결과 영상에 대해 스케일링 수행 후 임계화를 수행하여 문서의 내용이 명확하게 표시되도록 처리한다.
  - cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
  - cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
  - cv2.adaptiveThreshold(dst, 255, ..., 21, 10)
```

- 실행 결과
<img align="left" width="320" height="370" src="https://user-images.githubusercontent.com/74342121/180010708-85ca53ff-4aaf-4611-88de-c27a70ab9261.png">

<img align="left" width="320" height="370" src="https://user-images.githubusercontent.com/74342121/180010766-b8df2885-6f7a-43fe-ae8d-e6493a1c9aba.png">

<img align="center" width="320" height="370" src="https://user-images.githubusercontent.com/74342121/180010812-51547806-1af6-4f3c-9c7b-4dbceb4b6077.png">

<br/><br/><hr/>

### 3. Barcode detection(바코드 검출)
- 개요
```
Image filtering, Image segmentation, Morphological transformation, Connected component 기법을 활용한 바코드 영역 검출
```

- 문제 해결 과정
```
1. 입력 영상인 color 영상들을 grayscale 영상으로 변환한다.

2. 수평 및 수직 방향으로 경계 강도를 계산한다.

3. 수평 및 수직 방향을 기준으로 바코드 후보 영역을 검출한다.
  1) 수평 방향인 경우: 수평 방향의 경계 강도 – 수직 방향의 경계 강도
     수직 방향인 경우: 수직 방향의 경계 강도 – 수평 방향의 경계 강도

  2) 영상에 포한된 노이즈를 제거하기 위해 영상 평활화를 수행한다.

  3) 임계화(Thresholding)를 수행한다. 임계값은 적절한 값을 직접 지정한다.

  4) 모폴로지 변환을 수행한다. 수평 및 수직 방향으로 긴 사각형 모양을 갖는 구조적 요소를 사용하여 닫힘 연산을 수행한다.

  5) 작은 사각형 모양을 갖는 구조적 요소를 사용하여 침식과 팽창 연산을 반복하여 적용한다.

  6) 연결 요소를 생성한 후 가장 크기가 큰 한 개의 연결 요소를 선택한다.

  7) 수평 및 수직 방향의 처리 과정에서 선택한 두 개의 연결 요소의 크기를 비교한다. 
     이 중에서 크기가 큰 연결 요소를 최종 바코드 영역으로 검출한다.

4. 원본 영상에 검출한 바코드 영역을 표시한 후 결과 영상을 저장한다.

5. 지정한 폴더에 포함된 모든 영상에 대해 1~4단계를 반복한다.
```

- 실행 결과

<img align="center" width="450" height="250" src="https://user-images.githubusercontent.com/74342121/180015580-872f6636-8410-4db1-92ea-9faa02eada09.png">&nbsp;&nbsp;&nbsp;👉&nbsp;&nbsp;&nbsp;<img align="center" height="250" width="170" src="https://user-images.githubusercontent.com/74342121/180015857-dc0662fa-03c0-4245-9c4f-77b212f28512.jpg">

<img align="center" width="450" height="250" src="https://user-images.githubusercontent.com/74342121/180017975-b9830319-ae89-41bc-8d27-970a44ddfc4a.png">&nbsp;&nbsp;&nbsp;👉&nbsp;&nbsp;&nbsp;<img align="center" height="250" width="170" src="https://user-images.githubusercontent.com/74342121/180018277-ebaea995-7e45-47da-8bc3-ec2df2f1ecd8.jpg">

<img align="center" width="450" height="250" src="https://user-images.githubusercontent.com/74342121/180018488-82a1c7ff-fb15-4c25-9d59-543ef23b97b4.png">&nbsp;&nbsp;&nbsp;👉&nbsp;&nbsp;&nbsp;<img align="center" height="250" width="170" src="https://user-images.githubusercontent.com/74342121/180018620-7a1977b4-ddbc-4709-9d80-604c0d3f7a7f.jpg">

<br/><br/><hr/>

### 4. Car license plate detection(자동차 번호판 검출)
- 개요
```
자동차 번호판을 포함한 차량 영상으로부터 번호판 영역만을 분리하여 검출
```

- 문제 해결 과정
```
1. 결과 영상의 저장 폴더 존재 여부를 확인하며 없는 경우 저장 폴더를 생성한다. 그 후, 검출 결과 위치를 저장하기 위한 파일을 생성한다.

2. 불러온 color 영상을 grayscale 영상으로 변환한다.

3. detectCLP 함수를 통해 번호판 영역을 검출한다.
  1) 영상의 전처리를 진행한다. 전처리 진행과정은 아래와 같다.
  👉 GaussianBlur(영상 평활화) → AdaptiveThreshold(적응형 임계화) → Canny(에지 검출) → morhologyEx(모폴로지 연산(closing))

    1.1) 영상에 포함된 노이즈를 제거하기 위해 영상 평활화를 수행한다.

    1.2) 적응형 임계화(Thresholding)를 수행한다. 임계값은 적절한 값을 직접 지정한다.

    1.3) 영상의 에지 검출을 위해 Canny 연산을 수행한다.

    1.4) 모폴로지 변환을 수행한다. 긴 직사각형 모양을 갖는 구조적 요소를 사용하여 닫힘 연산을 수행한다. 
         작은 사각형 모양을 갖는 구조적 요소를 사용하여 팽창과 침식 연산을 적용한다.

  2) 연결 요소를 생성한 후 크기가 큰 상위 15개의 연결 요소를 선택한다.

  3) 선택한 연결 요소들 중에서 approxPolyDP() 함수를 사용하여 모양을 근사화한다.

  4) 근사화를 통해 나온 값을 boundingRect() 함수를 사용하여 윤곽선의 경계면을 
     둘러싸는 사각형을 계산하여 4개의 좌표(sx, sy, ex, ey)로 저장하도록 한다.

  5) 저장된 좌표를 통해 남은 bounding box 중 자동차 번호판 크기와 비슷한 비율을 찾기 위해 가로 세로 비율과 영역의 크기의 조건을 추가한다.

  6) 조건에 해당하는 사각형 좌표(sx, sy ex, ey) 값을 반환한다.

5. 검출한 번호판 영역을 표시한 후 결과 영상을 저장한다.

6. 저장을 모두 마친 경우, 타이머를 종료시키고 자동차 번호판을 검출하는 데 소요된 시간을 출력한다.
```

- 실행 결과

<img align="left" width="480" height="250" src="https://user-images.githubusercontent.com/74342121/180020919-dcea04cd-355a-4632-aac7-6a966fcf7c96.png">

<img align="center" width="480" height="250" src="https://user-images.githubusercontent.com/74342121/180021038-fe805673-2cb9-42ed-8b5e-1d990189a1ed.png">

<img align="left" width="480" height="250" src="https://user-images.githubusercontent.com/74342121/180021137-35ad58f9-2c2f-4ade-b13b-684785822efe.png">

<img align="center" width="480" height="250" src="https://user-images.githubusercontent.com/74342121/180022152-9b51120d-ee0a-4b4b-9fe1-9bc93322a242.png">

