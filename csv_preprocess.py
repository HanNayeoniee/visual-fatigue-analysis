import os
import csv
import numpy as np
import pandas as pd


"""
csv 보간하고 새로운 csv 만드는 코드
"""

# 평균1/3이하인 값들 0으로
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


# 앞/뒤 프레임 사이에 0프레임이 하나일때 앞/뒤 프레임의 평균
def interpolation_zero(y):
    for i in range(1, len(y) - 1):
        if y[i] == 0:
            if (y[i - 1] != 0) and (y[i + 1] != 0):
                y[i] = (y[i - 1] + y[i + 1]) // 2

    return y


# 앞/뒤 프레임이 0이고, 그 사이 값이 0이 아닐 때 0으로
def interpolation_value(y):
    for i in range(1, len(y) - 1):
        if y[i] != 0:
            if (y[i - 1] == 0) and (y[i + 1] == 0):
                y[i] = 0

    return y


def main():
    # path = './csv수정'
    path = '../[csvs]2세대 마스크 데이터-수정완료2'

    file_list = os.listdir(path)
    file_list_csv = [file for file in file_list if file.endswith(".csv")]

    for i, file in enumerate(file_list_csv):
        csv_dir = os.path.join(path, file)
        csv_file = pd.read_csv(csv_dir)
        pupil_list = csv_file['pupil_size_diameter'].tolist()

        interpol_zero = interpolation_zero(pupil_list)
        interpol_val = interpolation_value(interpol_zero)
        thresh_list = thres_zero(interpol_val)

        # 보간, 수정해서 csv로 뿌리기
        name = file.split('.')[0]
        with open(f'new_csv/{name}.csv', 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'pupil_size_diameter'])
            for i, a in enumerate(thresh_list):
                writer.writerow([i, a])


if __name__ == '__main__':
    main()
