import os
import cv2
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic


# 이미지를 읽어서 pyqt로 보여주는 함수
def cvtPixmap(frame, img_size):
    frame = cv2.resize(frame, img_size)
    height, width, channel = frame.shape
    bytesPerLine = 3 * width
    qImg = QImage(frame.data,
                  width,
                  height,
                  bytesPerLine,
                  QImage.Format_RGB888).rgbSwapped()
    qpixmap = QPixmap.fromImage(qImg)

    return qpixmap


# 동공 주변 반사광 채우는 함수
def fill_reflected_light(ori_img, min_thr, iteration=2, add_inter_idx=1):
    if len(ori_img.shape) == 3:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(ori_img, min_thr, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    img_thresh = cv2.dilate(img_thresh, kernel, iterations=iteration)  # 팽창연산-대상을 확장한 후 작은 구멍을 채우는 방식
    draw_img = ori_img.copy()  # 원본 이미지 복사

    reflection_points = np.where(img_thresh == 255)  # 화소값이 255인 인덱스 반환
    for y, x in zip(reflection_points[0], reflection_points[1]):
        # x 픽설의 왼쪽 픽셀이 l_x, 오른쪽 픽셀이 r_x
        # l_x는 0이상의 값을 가지고, r_x는 이미지크기보다 작아야 함
        l_x, r_x = x - 1, x + 1
        l_x = l_x if l_x >= 0 else 0
        r_x = r_x if r_x < img_thresh.shape[1] else img_thresh.shape[1] - 1
        
        # l_x, r_x가 이미지크기 범위 안에있고, 반사광이면 1칸씩 이동
        while l_x >= 0 and img_thresh[y][l_x] == 255:
            l_x -= 1
        while r_x < (img_thresh.shape[1] - 1) and img_thresh[y][r_x] == 255:
            r_x += 1

        # 반사광에서 가장 인접한 값이 아닌, 조금 옆의 값으로 반사광 채우기
        # 이미 위에서 dilation 연산을 통해 경계가 두꺼워져 반사광 조금 옆의 값을 가져왔기 때문에 add_inter_idx의 큰 의미는 없음
        l_x -= add_inter_idx
        r_x += add_inter_idx
        l_x = l_x if l_x >= 0 else 0
        r_x = r_x if r_x < img_thresh.shape[1] else img_thresh.shape[1] - 1

        l_val = int(ori_img[y][l_x])
        r_val = int(ori_img[y][r_x])
        draw_img[y][x] = int((l_val + r_val) / 2)  # 반사광 채우기
    return draw_img


# 동공 검출 함수
def getPupil(img, thresh, area_val, symmetry_val, fill_cond_val):
    '''
    :param img: 입력 동공 이미지
    :param thresh:
    :param area_val:
    :param symmetry_val:
    :param fill_cond_val:
    :return:
    condition으로 끝나는 변수 3개가 모두 1로 만족해야 res에 append할 수 있음
    area_condition : 직사각형 contour로 둘러싸인 부분의 면적
    symmetry_condition : 1-종횡비율(contour를 둘러싼 직사각형의 너비/높이 비율)이 symmetry_val(0.2)보다 작으면 1-> 정사각형에 가까워야함
    fill_condition : (contour로 둘러싸인 면적인 area/위에서 계산한 내접원의 넓이)을 계산해 얼마나 채워져있는지 비교
    '''
    res = []

    if len(img.shape) == 3:
        gray = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    ret, thresh_gray = cv2.threshold(gray, thresh[0], thresh[1], cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    draw_img = img.copy()
    for i in range(len(contours)):
        # # 컨투어 각각 시각화
        # cv2.drawContours(draw_img, [contours[i]], 0, (0, 0, 255), 2)
        # cv2.putText(draw_img, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        # print(i, hierarchy[0][i])
        # cv2.imshow('contour detection', draw_img)
        # cv2.waitKey(0)

    # for contour in contours:
        area = cv2.contourArea(contours[i])
        rect = cv2.boundingRect(contours[i])
        x, y, width, height = rect  # 직사각형 모양 바운딩 박스의 좌표, 너비, 높이
        radius = 0.25 * (width + height)  # 내접원의 반지름(내 생각엔 직사각형 모양의 contour의 내접원의 반지름같음)

        area_condition = (area_val <= area)
        symmetry_condition = (abs(1 - float(width) / float(height)) <= symmetry_val/10)
        fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= fill_cond_val/10)

        # 3가지 조건을 모두 만족해야 동공 영역
        if area_condition and symmetry_condition and fill_condition:
            res.append(((int(x + radius), int(y + radius)), int(1 * radius), rect))  # 동공중심 x좌표, y좌표, 반지름, rect(외접 사각형)

    return res, thresh_gray

# 동공 지름 구하기
def get_pupil_size(roi, binary_eye, pupil_info, add_radius):
    info = pupil_info[0]  # 동공중심 (x좌표, y좌표), 반지름, rect(외접 사각형)
    rect_roi = info[2]
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


def frames_to_timecode(total_frames, frame_rate=30, drop=False):
    """
    Method that converts frames to SMPTE timecode.

    :param total_frames: Number of frames
    :param frame_rate: frames per second
    :param drop: true if time code should drop frames, false if not
    :returns: SMPTE timecode as string, e.g. '01:02:12:32' or '01:02:12;32'
    """
    if drop and frame_rate not in [29.97, 59.94]:
        raise NotImplementedError("Time code calculation logic only supports drop frame "
                                  "calculations for 29.97 and 59.94 fps.")

    # for a good discussion around time codes and sample code, see
    # http://andrewduncan.net/timecodes/

    # round fps to the nearest integer
    # note that for frame rates such as 29.97 or 59.94,
    # we treat them as 30 and 60 when converting to time code
    # then, in some cases we 'compensate' by adding 'drop frames',
    # e.g. jump in the time code at certain points to make sure that
    # the time code calculations are roughly right.
    #
    # for a good explanation, see
    # https://documentation.apple.com/en/finalcutpro/usermanual/index.html#chapter=D%26section=6
    fps_int = int(round(frame_rate))

    if drop:
        # drop-frame-mode
        # add two 'fake' frames every minute but not every 10 minutes
        #
        # example at the one minute mark:
        #
        # frame: 1795 non-drop: 00:00:59:25 drop: 00:00:59;25
        # frame: 1796 non-drop: 00:00:59:26 drop: 00:00:59;26
        # frame: 1797 non-drop: 00:00:59:27 drop: 00:00:59;27
        # frame: 1798 non-drop: 00:00:59:28 drop: 00:00:59;28
        # frame: 1799 non-drop: 00:00:59:29 drop: 00:00:59;29
        # frame: 1800 non-drop: 00:01:00:00 drop: 00:01:00;02
        # frame: 1801 non-drop: 00:01:00:01 drop: 00:01:00;03
        # frame: 1802 non-drop: 00:01:00:02 drop: 00:01:00;04
        # frame: 1803 non-drop: 00:01:00:03 drop: 00:01:00;05
        # frame: 1804 non-drop: 00:01:00:04 drop: 00:01:00;06
        # frame: 1805 non-drop: 00:01:00:05 drop: 00:01:00;07
        #
        # example at the ten minute mark:
        #ㄺ
        # frame: 17977 non-drop: 00:09:59:07 drop: 00:09:59;25
        # frame: 17978 non-drop: 00:09:59:08 drop: 00:09:59;26
        # frame: 17979 non-drop: 00:09:59:09 drop: 00:09:59;27
        # frame: 17980 non-drop: 00:09:59:10 drop: 00:09:59;28
        # frame: 17981 non-drop: 00:09:59:11 drop: 00:09:59;29
        # frame: 17982 non-drop: 00:09:59:12 drop: 00:10:00;00
        # frame: 17983 non-drop: 00:09:59:13 drop: 00:10:00;01
        # frame: 17984 non-drop: 00:09:59:14 drop: 00:10:00;02
        # frame: 17985 non-drop: 00:09:59:15 drop: 00:10:00;03
        # frame: 17986 non-drop: 00:09:59:16 drop: 00:10:00;04
        # frame: 17987 non-drop: 00:09:59:17 drop: 00:10:00;05

        # calculate number of drop frames for a 29.97 std NTSC
        # workflow. Here there are 30*60 = 1800 frames in one
        # minute

        # 2는 왜 뺴지?
        FRAMES_IN_ONE_MINUTE = 1800 - 2
        FRAMES_IN_TEN_MINUTES = (FRAMES_IN_ONE_MINUTE * 10) - 2

        ten_minute_chunks = total_frames / FRAMES_IN_TEN_MINUTES  # 10분짜리가 몇 묶음인지
        one_minute_chunks = total_frames % FRAMES_IN_TEN_MINUTES  # 1분짜리가 몇 묶음인지

        ten_minute_part = 18 * ten_minute_chunks
        one_minute_part = 2 * ((one_minute_chunks - 2) / FRAMES_IN_ONE_MINUTE)

        if one_minute_part < 0:
            one_minute_part = 0

        # add extra frames
        total_frames += ten_minute_part + one_minute_part

        # for 60 fps drop frame calculations, we add twice the number of frames
        if fps_int == 60:
            total_frames = total_frames * 2

        # time codes are on the form 12:12:12;12
        smpte_token = ";"

    else:
        # time codes are on the form 12:12:12:12
        smpte_token = ":"

    # now split our frames into time code
    hours = int(total_frames / (3600 * fps_int))
    minutes = int(total_frames / (60 * fps_int) % 60)
    seconds = int(total_frames / fps_int % 60)
    frames = int(total_frames % fps_int)
    return "%02d:%02d:%02d%s%02d" % (hours, minutes, seconds, smpte_token, frames)
