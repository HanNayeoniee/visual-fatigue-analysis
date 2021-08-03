from analysis_utils import *
import openpyxl

"""
csv_preprocess.py를 실행해 프레임별 동공 크기가 들어있는 csv파일을 보간한 후 analysis.py 실행하기
눈깜빡임 횟수, 눈 감은 시간, 동공 크기 변화율 분석하는 코드
"""

def main():
    # path : 동공크기가 들어있는 엑셀파일 경로
    # path = './new_csv'  # 28명 전체
    # path = './new_csv_10_blink_best'  # 눈 깜빡임 잘나온 10명
    path = './new_csv_10_best_0629'  # 동공변화속도 & 눈감/눈깜 잘나온 10명
    # path = './new_csv_10_size_best'  # 동공변화속도 잘나온 10명

    # 분석 결과 이미지를 저장할 경로
    save_img_blink = './img_blink_0628'
    save_img_close = './img_close_0628'
    save_img_size = './img_size_0628'
    save_img_fft = './img_fft_0628'  # zero-crossing그래프 저장
    minute = None
    quarter = 10

    if quarter is not None:
        chunk = quarter
    else:
        chunk = 30 // minute

    file_list = os.listdir(path)
    file_list_csv = [file for file in file_list if file.endswith(".csv")]

    avg_blink = []
    avg_frame = []
    avg_size = []

    for i, file in enumerate(file_list_csv):
        name = file.split(".")[0]  # 파일명
        csv_dir = os.path.join(path, file)
        csv_file = pd.read_csv(csv_dir)
        pupil_list = csv_file['pupil_size_diameter'].tolist()  # 동공 크기 데이터만 가져오기

        # 1) 눈 감은시간, 눈 깜빡임 횟수 분석
        pupil_frames, pupil_blinks = count_blink(pupil_list, minu=minute, quar=quarter)  # 엑셀파일 한사람씩 불러오기

        # 분석 결과 사람별 엑셀파일로 저장 - 정규화 안하는 경우
        BD_df = pd.DataFrame(pupil_frames)
        BF_df = pd.DataFrame(pupil_blinks)
        BD_df.to_csv('./res_blink_duration/' + name + '.csv', index=True, encoding='cp949')
        BF_df.to_csv('./res_blink_frequency/' + name + '.csv', index=True, encoding='cp949')
        print('done' + name)

        # # 눈 감은시간, 눈 깜빡임 횟수 정규화
        # pupil_frame = np.array(pupil_frames)
        # pupil_blink = np.array(pupil_blinks)
        # frame_norm = []
        # blink_norm = []
        # sum_f = np.sum(pupil_frame)
        # sum_b = np.sum(pupil_blink)
        # for f_value, b_value in zip(pupil_frame, pupil_blink):
        #     f_norm = f_value / sum_f
        #     b_norm = b_value / sum_b
        #     frame_norm.append(f_norm)
        #     blink_norm.append(b_norm)
        # print(len(frame_norm))
        # print(len(blink_norm))

        # # 분석 결과 사람별 엑셀파일로 저장 - 정규화 하는 경우
        # BD_df = pd.DataFrame(frame_norm)
        # BF_df = pd.DataFrame(blink_norm)
        # BD_df.to_csv('./norm_res_blink_duration_0629/' + name + '.csv', index=True, encoding='cp949')
        # BF_df.to_csv('./norm_res_blink_frequency_0629/' + name + '.csv', index=True, encoding='cp949')
        # print('done' + name)


        
        
        # # 눈 감은 시간, 눈깜빡임 횟수 막대 그래프 그리기
        # draw_graph_poly(frame_norm, file, 0.3, '눈감은 시간', f'{save_img_close}/{name}_눈감.png', quar=quarter)
        # draw_graph_poly(blink_norm, file, 0.3, '눈깜빡임 횟수', f"{save_img_blink}/{name}_눈깜.png", quar=quarter)
        # print("done")

        # 2) 동공크기 변화율 분석
        # # 동공크기 변화율
        filter, zero_cross, section_frames, change_rates_list = fft(pupil_list, minu=minute, quar=quarter)
        draw_fft_graph(pupil_list, filter, zero_cross, section_frames, f'{save_img_fft}/{name}.png')

        # # 분기 별 변화율
        # change_rates = []
        # for ii, change_rate in enumerate(change_rates_list):
        #     if change_rate:
        #         average_change_rate = sum(change_rate) / len(change_rate)
        #         change_rates.append(average_change_rate)

        # # 동공 크기 변화율 그래프 그리기
        # draw_graph_poly(change_rates, name, 0.06, '동공크기 변화율', f'{save_img_size}/{name}_동공변화.png', quar=quarter)
        # print(i, "done")

    #     # 사람간 평균 (분기수가 같을때만)
    #     if len(frame_norm) == chunk:
    #         avg_frame.append(frame_norm)
    #         avg_blink.append(blink_norm)
    #         avg_size.append(change_rates)
    #
    # avg_frame = np.array(avg_frame)
    # avg_blink = np.array(avg_blink)
    # avg_size = np.array(avg_size)
    # frame = avg_frame.mean(axis=0)
    # blink = avg_blink.mean(axis=0)
    # size = avg_size.mean(axis=0)

    # # 평균 그래프 출력
    # draw_graph_poly(frame, '눈감은시간 10명 평균', 0.15, '눈감은시간', f'{save_img_close}/[평균10]_눈감은시간.png', quar=quarter)
    # print("done1")
    # draw_graph_poly(blink, '눈깜빡임횟수 10명 평균', 0.15, '눈깜빡임 횟수', f'{save_img_blink}/[평균10]_눈깜빡임횟수.png', quar=quarter)
    # print("done2")
    # draw_graph_poly(size, '동공크기 변화율 10명 평균', 0.03, '동공크기 변화율', f'{save_img_size}/[평균]_동공크기 변화율.png', quar=quarter)
    # print("done3")

if __name__ == '__main__':
    main()