import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack, signal
from scipy.optimize import curve_fit

# 고주파 성분만 날리는 fft
def get_filtered_data(in_data, filter_value=0.004):
    """
    :param in_data: 대상 시계열 신호 
    :param filter_value: filter_value이상의 주파수를 가지는 신호를 날림 
    :return: fft 결과
    """
    sig_fft = fftpack.fft(in_data)
    sample_freq = fftpack.fftfreq(in_data.size)
    high_freq_fft = sig_fft.copy()

    high_freq_fft[np.abs(sample_freq) > filter_value] = 0
    filtered_data = fftpack.ifft(high_freq_fft)

    return filtered_data

# 고주파, 저주파 성분을 날리는 fft
def get_filtered_data1(in_data, high_filter_value, low_filter_value):
    """
    :param in_data: 대상 시계열 신호 
    :param high_filter_value: fft를 수행할 최대값, low_filter_value ~ high_filter_value값 사이의 신호를 fft
    :param low_filter_value: fft를 수행할 최소값
    :return: fft 결과
    """
    sig_fft = fftpack.fft(in_data)
    sample_freq = fftpack.fftfreq(in_data.size)
    high_freq_fft = sig_fft.copy()

    low_value1 = np.max(high_freq_fft)
    high_freq_fft[np.abs(sample_freq) > high_filter_value] = 0
    high_freq_fft[np.abs(sample_freq) < low_filter_value] = 0

    low_value2 = np.max(high_freq_fft)
    filtered_data = fftpack.ifft(high_freq_fft)

    return filtered_data, low_value1, low_value2

# fft 건하.ver
def filter_bandpass(arr, srate, band):
    """
    :param arr: 대상 시계열 신호
    :param srate: 샘플링율(=프레임율)
    :param band: 통과 대역(tuple 자료형)
    :return:
    """
    try:
        nyq = 60 * srate / 2
        coef_vector = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
        return signal.filtfilt(*coef_vector, arr)
    except ValueError:
        return arr


# 눈 깜빡임 시간(프레임수), 횟수 세기
def count_blink(pupil_list, minu=None, quar=None):
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

    return pupil_frame, pupil_blink


# 엑셀 대신 바로 여기서 그래프 저장하기
# 눈 깜빡임 시간, 횟수 그래프를 그릴 때 사용
def draw_graph(norm_value, title, y_lim, y_label, savepath, quar=None):
    if quar is None:
        period = ['0~3분', '3~6분', '6~9분', '9~12분', '12~15분', '15~18분', '18~21분', '21~24분', '24~27분', '27~30분', '30~33분']
    else:
        period = [i+1 for i in range(quar)]
    plt.rcParams["font.family"] = 'Malgun Gothic'

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    for i, x in enumerate(norm_value):
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
    # plt.savefig(f'{savepath}')
    plt.imshow(img)

# 2차식 추세선 그리기, 히스토그램 그래프 저장
# 추세식 : y = a*x^2 + b*x + c
def draw_graph_poly(norm_value, title, y_lim, y_label, savepath, quar=None):
    results = {}
    # y = x^2 + 4x + 4 = (x+2)^2
    # x = np.array([1, 2, 3])
    # y = np.array([9, 16, 25])

    # 추세선
    x = np.arange(0, len(norm_value))  # (1, 2, 3, ... 10)
    y = []
    for idx, value in enumerate(norm_value):
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
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['r-squared'] = ssreg / sstot
    r_squared = str(round(results['r-squared'], 3))  # 출력하기 위해 문자열로 변환
    a = str(round(results['coeffs'][0], 3))  
    b = str(round(results['coeffs'][1], 3))
    c = str(round(results['coeffs'][2], 3))
    print("R 제곱값: ", round(results['r-squared'], 3))
    print("추세선: "+"Y="+a+"xX^2 + "+b+"xX + "+c)

    period = ['0~3분', '3~6분', '6~9분', '9~12분', '12~15분', '15~18분', '18~21분', '21~24분', '24~27분', '27~30분', '30~33분']
    plt.rcParams["font.family"] = 'Malgun Gothic'

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    for idx2, value2 in enumerate(norm_value):
        ax.bar(period[idx2], value2, color='b', alpha=0.5)
    ax.plot(x, fit_equation, color='r', alpha=0.5, label='Polynomial fit', linewidth=3.0)
    # ax.scatter(x, y, s = 5, color = 'b', label = 'Data points')  # 추세선 예측에 사용한 좌표 그리기

    # Plotting
    plt.xticks(rotation=20)
    plt.title(f'{title}')
    plt.ylim(0, y_lim)
    plt.xlabel('구간')
    plt.ylabel(f'{y_label}')
    # # 눈감, 눈깜 출력할 때 위치 조정
    # plt.text(3.2, 0.28, "추세선: "+r'$y = $' + a + r'$x^2 + ($'+ b + r'$)x + $' + c, fontdict={'size': 12})
    # plt.text(7.5, 0.26, r'$R^2 =$'+r_squared, fontdict={'size': 12})

    # # 동공 크기 변화율 출력할 때 위치 조정
    # plt.text(3.2, 0.055, "추세선: " + r'$y = $' + a + r'$x^2 + ($' + b + r'$)x + $' + c, fontdict={'size': 12})
    # # plt.text(7.5, 0.05, r'$R^2 =$' + r_squared, fontdict={'size': 12})

    # # 평균 눈감, 눈깜 출력할 때 위치 조정
    # plt.text(3.2, 0.14, "추세선: " + r'$y = $' + a + r'$x^2 + ($' + b + r'$)x + $' + c, fontdict={'size': 12})
    # plt.text(7.5, 0.12, r'$R^2 =$' + r_squared, fontdict={'size': 12})

    # 평균 동공크기 변화율 출력할 때 위치 조정
    plt.text(3.2, 0.027, "추세선: " + r'$y = $' + a + r'$x^2 + ($' + b + r'$)x + $' + c, fontdict={'size': 12})
    plt.text(7.5, 0.025, r'$R^2 =$' + r_squared, fontdict={'size': 12})
    plt.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    spl = title.split('.')[0]
    plt.savefig(f'{savepath}')
    plt.imshow(img)
    plt.show()  # 그래프 잘 나오는지 띄우기


# 로그함수 추세선 그리기
# 추세식 : y = a*log(x) + b
# reference 1 : https://stats.stackexchange.com/questions/190107/curve-fit-with-logarithmic-regression-in-python
# reference 2 : https://turtle-dennis.tistory.com/17
# reference 3 : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# def log_func(x, p1, p2):
#     return p1*np.log(x)+p2
#
# # 데이터에 맞는 로그함수를 찾을 수 없으면 에러남(2세대 마스크에서는 사용X)
# def draw_graph_log(norm_value, title, y_lim, y_label, savepath, quar=None):
#     x_data = np.array([10, 100, 1000])
#     y_data = np.array([1, 2, 3])
#     ppot, pcov = curve_fit(log_func, x_data, y_data)  # ppot는 찾고자하는 함수의 파라미터, pcov는 공분산 추정값
#
#     p1 = ppot[0]
#     p2 = ppot[1]
#
#     # plot curve
#     curvex=np.linspace(0,1000,1000)  # 균등한 간격을 가진 시퀀스 생성
#     curvey=log_func(curvex,p1,p2)
#     plt.plot(curvex, curvey,'r', linewidth=3)  # 로그함수모양 곡선으로 그리기
#     plt.plot(x_data, y_data, 'o', label='original')
#     plt.plot(x_data, log_func(x_data, p1, p2), label='fitting')  # 직선으로 그리기
#     plt.legend()


# get_filtered_data1()함수로 fft 수행
def fft(pupil_list, minu=None, quar=None):
    global section_frames, time
    # 데이터에서 0, -1인 부분 제거
    while 0 in pupil_list:
        pupil_list.remove(0)
    while -1 in pupil_list:
        pupil_list.remove(-1)

    if minu is not None:
        time = minu * 1800
        section_frames = len(pupil_list) // time

    if quar is not None:
        time = len(pupil_list) // quar
        section_frames = quar

    y = np.array(pupil_list)

    # fft
    # filtered_sig = get_filtered_data(y, filter_value=0.005)  # 태형.ver(고주파 날리기)
    # filtered_sig = filter_bandpass(y, 300, [500, 800])  # 건하.ver(결과 별로임)
    filtered_sig, _, _ = get_filtered_data1(y, 0.0048, 0.0035)  # 경원.ver(저주파, 고주파 날리기)
    filtered_sig = filtered_sig.astype(np.float)

    # zero-crossing point
    zero_crossings = np.where(np.diff(np.sign(np.diff(filtered_sig))))[0]
    zero_crossings = np.insert(zero_crossings, 0, 0)
    zero_crossings = np.append(zero_crossings, len(filtered_sig) - 1)

    # 변화 속도 계산
    change_rates_list = [[] for _ in range(section_frames)]
    for section in range(section_frames):
        # zero-crossing points 기준으로 원하는 위치(섹션) 가져오기
        section_zero_crossing = zero_crossings[np.where(zero_crossings <= (section + 1) * time)]
        section_zero_crossing = section_zero_crossing[np.where(section * time < section_zero_crossing)]
        # 변화 속도 계산
        for j in range(len(section_zero_crossing) - 1):
            change_rate = abs((filtered_sig[section_zero_crossing[j + 1]] - filtered_sig[section_zero_crossing[j]]) / (
                        section_zero_crossing[j + 1] - section_zero_crossing[j]))
            change_rates_list[section].append(change_rate)

    return filtered_sig, zero_crossings, section_frames, change_rates_list


# fft를 수행한 결과 그래프 그리기
def draw_fft_graph(y, filtered_sig, zero_crossings, section_frames, savepath, minu=None, quar=None):
    global time
    x = np.arange(0, len(y))
    if minu is not None:
        time = minu * 1800
        section_frames = len(y) // time

    if quar is not None:
        time = len(y) // quar
        section_frames = quar

    fig = plt.figure(dpi=150)

    # plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = 'Malgun Gothic'
    plt.figure(figsize=(14, 6))
    plt.plot(x, y, label='Original signal')
    plt.plot(x, filtered_sig, linewidth=2, label='Filtered signal')
    # plt.plot(zero_crossings, filtered_sig[zero_crossings], marker='o', color='red', linestyle='--')
    plt.legend(loc='upper right')

    # 섹션 나눠진거 표시
    for section in range(section_frames):
        plt.axvline(x=section * time, ymin=0, ymax=1.0, color='r')
        plt.axvline(x=(section + 1) * time, ymin=0, ymax=1.0, color='r')

    # plt.xlim(0, 1800)
    plt.title('동공크기 변화율')
    plt.xlabel('Frame')
    plt.ylabel('Pupil size')
    plt.savefig(f'{savepath}')
    plt.show()

