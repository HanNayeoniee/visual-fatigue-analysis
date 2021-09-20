import cv2
import math
import numpy as np

# getPupil()함수 설명
def getPupil(img, thresh):
    res = []

    original = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)  # 입력으로 들어온 img를 grayscale로 변환

    # 이진화: 이미지가 thresh[0]보다 작으면 0, 크면 thresh[1] 할당, thresh_gray는 이진화한 이미지
    thresh = [215, 255]
    ret, thresh_gray = cv2.threshold(gray, thresh[0], thresh[1], cv2.THRESH_BINARY)

    # contours에는 컨투어가 리스트로 들어있음
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)  # 폐곡선인 contour로 둘러싸인 부분의 면적
        rect = cv2.boundingRect(contour)  # contour라인을 둘러싸는 사각형 그리기
        x, y, width, height = rect  # 직사각형 모양 바운딩 박스의 좌표, 너비, 높이
        radius = 0.25 * (width + height)  # 내접원의 반지름(내 생각엔 직사각형 모양의 contour의 내접원의 반지름같음)

        # condition으로 끝나는 변수 3개가 모두 1로 만족해야 res에 append할 수 있음
        # area_condition : 직사각형 contour로 둘러싸인 부분의 면적이 100보다 큰지
        # symmetry_condition : 1-종횡비율(contour를 둘러싼 직사각형의 너비/높이 비율)이 0.2보다 작으면 1-> 정사각형에 가까워야함
        # fill_condition : (contour로 둘러싸인 면적/위에서 계산한 내접원의 넓이)을 계산해 얼마나 채워져있는지 비교
        area_condition = (70 <= area)
        symmetry_condition = (abs(1 - float(width) / float(height)) <= 0.2)  # (1-종횡비)가 0.2이하->정사각형에 가까워야함
        fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.3)

        # 3가지 조건 중 하나라도 만족하지 않으면 동공이 아닌 contour가 나옴
        if area_condition and symmetry_condition and fill_condition:
            res.append(((int(x + radius), int(y + radius)), int(1 * radius)))  # 동공중심의 x좌표, y좌표, 반지름


    cv2.drawContours(gray, contours, -1, (0,0,255), 1)
    cv2.imshow('contour', gray)
    # bbox = cv2.circle(gray, (x, y), int(radius), (0, 255, 0), 3)
    # cv2.imshow('bbox', bbox)
    # circle = cv2.circle(original, (x, y), int(radius), (255, 0, 0), -1)
    # cv2.imshow('circle', circle)
    return res, thresh_gray


# main
cap = cv2.VideoCapture('./video/sample1.avi')


while True:
    ret, frame = cap.read()

    if ret:
        res, thresh_gray = getPupil(frame, [180, 255])

        cv2.imshow('binarization', thresh_gray)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()