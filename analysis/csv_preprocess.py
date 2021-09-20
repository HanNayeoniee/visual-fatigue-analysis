import os
import csv
import numpy as np
import pandas as pd


"""
csv 보간하고 새로운 csv 만드는 코드
"""

# 폴더 생성
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!", directory)
            raise


# 앞/뒤 프레임 사이에 0프레임이 하나일때 앞/뒤 프레임의 평균으로 대체
# frame 1, 2, 3의 크기가 순서대로 a, 0, b일 때 frame 2 = (a+b) // 2
def interpolation_zero(y):
    for i in range(1, len(y) - 1):
        if y[i] == 0:
            if (y[i - 1] != 0) and (y[i + 1] != 0):
                y[i] = (y[i - 1] + y[i + 1]) // 2

    return y


# 앞/뒤 프레임이 모두 0이고, 그 사이 값이 0이 아닐 때 0으로 대체
# frame 1, 2, 3의 크기가 순서대로 0, a, 0일 때 frame 2 = 0
def interpolation_nonzero(y):
    for i in range(1, len(y) - 1):
        if y[i] != 0:
            if (y[i - 1] == 0) and (y[i + 1] == 0):
                y[i] = 0

    return y

# 평균의 1/3이하인 값들 0으로
def thres_zero(y):
    countzero = y.count(0)
    countminus = y.count(-1)
    minus = -1 * countminus
    num = len(y) - countzero - countminus

    avg_nonzero = (sum(y) - (-minus)) / num
    thres = int(avg_nonzero / 3)

    y = np.array(y)
    new_y = np.where(y < thres, np.where(y > 0, 0, y), y)

    return list(new_y)


def main():
    path = '../[csvs]2세대 마스크 데이터-수정완료2-new'

    file_list = os.listdir(path)
    file_list_csv = [file for file in file_list if file.endswith(".csv")]

    for i, file in enumerate(file_list_csv):
        csv_dir = os.path.join(path, file)
        csv_file = pd.read_csv(csv_dir)
        pupil_list = csv_file['pupil_size_diameter'].tolist()

        interpol_zero = interpolation_zero(pupil_list)
        interpol_nonzero = interpolation_nonzero(interpol_zero)
        thresh_list = thres_zero(interpol_nonzero)

        # 보간 후 csv로 저장
        name = file.split('.')[0]
        createFolder('./preprocess/')
        with open(f'preprocess/{name}.csv', 'a', encoding='utf-8-sig', newline='') as f:
            print("csv 전처리 완료:", f'preprocess/{name}.csv')
            writer = csv.writer(f)
            writer.writerow(['frame', 'pupil_size_diameter'])
            for i, a in enumerate(thresh_list):
                writer.writerow([i, a])


# if __name__ == '__main__':
#     main()
