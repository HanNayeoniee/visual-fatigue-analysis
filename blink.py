import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack, signal
from scipy.optimize import curve_fit



def count_blink(pupil_list, minu = None, quar = None, norm = False):
    """
    :param pupil_list: 프레임별 동공 크기가 들어있는 리스트
    :param minu: 전체 동공 데이터를 분 단위로 분석
    :param quar: 전체 동공 데이터를 몇개의 구간으로 나누어서 분석, minu 또는 quar 1개만 사용해야 함
    :return: 눈 깜빡임 시간(프레임수), 눈 깜빡임 횟수
    """

    # 전체 영상을 minu단위로 분석
    if minu is not None:
        time = minu * 1800  # 프레임수, 동공영상은 30fps
        chunk = len(pupil_list) // time  # 입력받은 csv파일이 몇개의 구간으로 나뉘는지

    # 전체 영상을 quar개로 쪼개서 분석
    if quar is not None:
        time = len(pupil_list) // quar
        chunk = quar

    pupil_frame = []
    pupil_blink = []

    if (chunk < 1):
        pupil_frame.append(-1)
        pupil_blink.append(-1)

    for i in range(chunk):
        # 눈 감은 프레임수 세기
        count_frame = pupil_list[i * time:(i + 1) * time].count(0)
        pupil_frame.append(count_frame)

        # 눈 감은 횟수 세기
        count = 0
        for idx in range(time * i, time * (i + 1)):
            if idx >= len(pupil_list):
                break
            if pupil_list[idx] == 0 and pupil_list[idx - 1] != 0:
                count += 1  # 눈 깜빡임 count
        pupil_blink.append(count)


    # 정규화 하는 경우
    if norm:
        pupil_frame = np.array(pupil_frame)
        pupil_blink = np.array(pupil_blink)
        frame_norm = []
        blink_norm = []
        sum_f = np.sum(pupil_frame)
        sum_b = np.sum(pupil_blink)
        for f_value, b_value in zip(pupil_frame, pupil_blink):
            f_norm = f_value / sum_f
            b_norm = b_value / sum_b
            frame_norm.append(f_norm)
            blink_norm.append(b_norm)

    # 정규화 안하는 경우는 그대로 반환
    return pupil_frame, pupil_blink, norm


# 그래프 그리고 바로 저장 - 눈 깜빡임 시간, 횟수
def draw_graph(data, title, y_lim, y_label, savepath, quar=None):
    if quar is None:
        period = ['0~3분', '3~6분', '6~9분', '9~12분', '12~15분', '15~18분', '18~21분', '21~24분', '24~27분', '27~30분', '30~33분']
    else:
        period = [i+1 for i in range(quar)]
    plt.rcParams["font.family"] = 'Malgun Gothic'

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    for i, x in enumerate(data):
        ax.bar(period[i], x, color='b', alpha=0.5)

    plt.xticks(rotation=20)
    plt.title(f'{title}')
    plt.ylim(0, y_lim)
    plt.xlabel('구간')
    plt.ylabel(f'{y_label}')
    plt.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    spl = title.split('.')[0]
    plt.savefig(f'{savepath}')
    plt.imshow(img)


# 2차식 추세선 그리기, 히스토그램 그래프 저장
# 추세식 : y = a*x^2 + b*x + c
def draw_trendline_blink(data, title, y_lim, y_label, savepath, quar = None, avg = False):
    """
    :param data: 그래프 그릴 데이터, numpy array 타입
    :param title: 그래프 제목
    :param y_lim:
    :param y_label:
    :param savepath: 그래프 저장경로
    :param quar:
    :param avg: 평균 그래프 시각화 여부
    :return:
    """
    results = {}

    # 추세선
    x = np.arange(0, len(data))
    y = []
    for idx, value in enumerate(data):
        y.append(value)
    y = np.array(y)  # 10개 구간에 해당하는 특징(깜빡임 횟수)

    fit = np.polyfit(x, y, 2)
    a = fit[0]
    b = fit[1]
    c = fit[2]
    fit_equation = a * np.square(x) + b * x + c
    results['coeffs'] = fit.tolist()

    # r-squared
    p = np.poly1d(fit)

    # fit values, and mean
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['r-squared'] = ssreg / sstot
    r_squared = str(round(results['r-squared'], 3))  # 출력하기 위해 문자열로 변환
    a = str(round(results['coeffs'][0], 3))
    b = str(round(results['coeffs'][1], 3))
    c = str(round(results['coeffs'][2], 3))
    # print("R 제곱값: ", round(results['r-squared'], 3))
    # print("추세선: "+"Y="+a+"xX^2 + "+b+"xX + "+c)

    period = ['0~3분', '3~6분', '6~9분', '9~12분', '12~15분', '15~18분', '18~21분', '21~24분', '24~27분', '27~30분', '30~33분']
    plt.rcParams["font.family"] = 'Malgun Gothic'

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    for idx2, value2 in enumerate(data):
        ax.bar(period[idx2], value2, color='b', alpha=0.5)
    ax.plot(x, fit_equation, color='r', alpha=0.5, label='Polynomial fit', linewidth=3.0)
    # ax.scatter(x, y, s = 5, color = 'b', label = 'Data points')  # 추세선 예측에 사용한 좌표 그리기

    # Plotting
    plt.xticks(rotation=20)
    plt.title(f'{title}')
    plt.ylim(0, y_lim)
    plt.xlabel('구간')
    plt.ylabel(f'{y_label}')

    # 눈감, 눈깜 출력할 때 위치 조정
    if not avg:
        plt.text(3.2, 0.28, "추세선: "+r'$y = $' + a + r'$x^2 + ($'+ b + r'$)x + $' + c, fontdict={'size': 12})
        plt.text(7.5, 0.26, r'$R^2 =$'+r_squared, fontdict={'size': 12})

    # 평균 눈감, 눈깜 출력할 때 위치 조정
    else:
        plt.text(3.2, 0.14, "추세선: " + r'$y = $' + a + r'$x^2 + ($' + b + r'$)x + $' + c, fontdict={'size': 12})
        plt.text(7.5, 0.12, r'$R^2 =$' + r_squared, fontdict={'size': 12})

    plt.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    spl = title.split('.')[0]
    plt.savefig(f'{savepath}')
    plt.imshow(img)
    plt.show()  # 그래프 잘 나오는지 띄우기
