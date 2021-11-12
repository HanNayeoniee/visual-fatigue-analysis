import cv2
import math
import numpy as np


def getPupil(img, thresh, area_val, symmetry_val, fill_cond_val):
    '''
    condition으로 끝나는 변수 3개가 모두 1로 만족해야 res에 append할 수 있음
    area_condition : 직사각형 contour로 둘러싸인 부분의 면적이 100보다 큰지
    symmetry_condition : 1-종횡비율(contour를 둘러싼 직사각형의 너비/높이 비율)이 0.2보다 작으면 1-> 정사각형에 가까워야함
    fill_condition : (contour로 둘러싸인 면적인 area/위에서 계산한 내접원의 넓이)을 계산해 얼마나 채워져있는지 비교
    '''

    res = []
    original = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)  # 입력으로 들어온 img를 grayscale로 변환
    draw_img = img.copy()

    # 이진화: 이미지가 thresh[0]보다 작으면 0, 크면 thresh[1] 할당, thresh_gray는 이진화한 이미지
    ret, thresh_gray = cv2.threshold(gray, thresh[0], thresh[1], cv2.THRESH_BINARY)

    # contours에는 컨투어가 리스트로 들어있음
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("total number of contours: ", len(contours))

    for i in range(len(contours)):
        # # 컨투어 각각 시각화
        # cv2.drawContours(draw_img, [contours[i]], 0, (0, 0, 255), 2)
        # cv2.putText(draw_img, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        # print(i, hierarchy[0][i])
        # cv2.imshow('contour detection', draw_img)
        # cv2.waitKey(0)


        area = cv2.contourArea(contours[i])  # 폐곡선인 contour로 둘러싸인 부분의 면적
        rect = cv2.boundingRect(contours[i])  # contour라인을 둘러싸는 사각형
        x, y, width, height = rect  # 직사각형 모양 바운딩 박스의 좌표, 너비, 높이
        radius = 0.25 * (width + height)  # 내접원의 반지름(내 생각엔 직사각형 모양의 contour의 내접원의 반지름같음)


        area_condition = (70 <= area)
        symmetry_condition = (abs(1 - float(width) / float(height)) <= 0.2)  # (1-종횡비)가 0.2이하->정사각형에 가까워야함
        fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.3)

        # 3가지 조건 중 하나라도 만족하지 않으면 동공이 아닌 contour가 나옴
        if area_condition and symmetry_condition and fill_condition:
            res.append(((int(x + radius), int(y + radius)), int(1 * radius), rect))  # 동공중심 x좌표, y좌표, 반지름, rect(외접 사각형)
            # res.append(((int(x + radius), int(y + radius)), int(1 * radius), rect))
            cv2.drawContours(gray, contours[i], -1, (0, 0, 255), 5)  # 조건에 맞는 동공 영역만 그리기

    # cv2.drawContours(gray, contours, -1, (0,0,255), 5)  # 전체 컨투어 그리기
    cv2.imshow('contour', gray)

    # bbox = cv2.circle(gray, (x, y), int(radius), (0, 255, 0), 3)
    # cv2.imshow('bbox', bbox)
    # circle = cv2.circle(original, (x, y), int(radius), (255, 0, 0), -1)
    # cv2.imshow('circle', circle)
    return res, thresh_gray


# 동공 지름 구하기
def get_pupil_size(roi, binary_eye, pupil_info, add_radius):
    info = pupil_info[0]  # (동공중심x,y), 반지름, (외접 사각형 x,y,w,h)
    # rect_roi = info[2]
    rect_roi = pupil_info[0][2]
    # print("pupil info[0]: ", info)
    # print("pupil info[1]: ", pupil_info[1])
    # print("pupil info[2]: ", rect_roi)

    box_x, box_y, width, height = rect_roi
    box_x = box_x - add_radius if box_x - add_radius >= 0 else 0
    box_y = box_y - add_radius if box_y - add_radius >= 0 else 0
    width = width + (2 * add_radius) if width + (2 * add_radius) <= roi.shape[1] else roi.shape[1]
    height = height + (2 * add_radius) if height + (2 * add_radius) <= roi.shape[0] else roi.shape[0]
    img_eye_only = binary_eye[box_y:box_y + height, box_x:box_x + width].copy()
    img_eye_only = np.where(img_eye_only == 255, 1, img_eye_only)

    cv2.rectangle(roi, (box_x, box_y), ((box_x + width), (box_y + height)), (0, 255, 255), 2)  # 동공주변 노란색 박스

    max_idx, max_val = 0, 0
    for col_idx in range(img_eye_only.shape[0]):
        col_val = sum(img_eye_only[col_idx])
        if max_val < col_val:
            max_idx = col_idx
            max_val = col_val

    l_row, r_row = 0, img_eye_only.shape[1]
    for row_idx in range(img_eye_only.shape[1] - 1):
        row_val = sum(img_eye_only[:, row_idx])
        if row_val != 0:
            l_row = row_idx
    for row_idx in range(img_eye_only.shape[1] - 1, 0, -1):
        row_val = sum(img_eye_only[:, row_idx])
        if row_val != 0:
            r_row = row_idx

    cv2.line(roi,
             (box_x + l_row, box_y + max_idx),
             (box_x + r_row, box_y + max_idx),
             (0, 0, 255), 2)  # 동공의 지름 그리기

    return roi, max_val



if __name__ == "__main__":
    # cap = cv2.VideoCapture('./video/sample1.avi')
    cap = cv2.VideoCapture('./video/jeontae.avi')

    while True:
        ret, frame = cap.read()

        if ret:
            pupil_info, binary_eye = getPupil(frame, thresh = [180, 255], area_val=70, symmetry_val=2, fill_cond_val=3)
            res, max_val = get_pupil_size(frame, binary_eye, pupil_info, 10)

            cv2.imshow('binarization', binary_eye)
            cv2.imshow('res', res)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()