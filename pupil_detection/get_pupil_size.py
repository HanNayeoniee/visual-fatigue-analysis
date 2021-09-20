# pyinstaller -w -F --icon=./graph2.ico get_pupil_size.py

# 2.8mm == 더 가까움
# 2.1mm == 더 멈

from utils import *
import csv


main_ui = uic.loadUiType('get_pupil_size.ui')[0]


class MyApp(QMainWindow, main_ui):
    def __init__(self):
        super(MyApp, self).__init__()
        # Window 초기화
        self.setupUi(self)
        self.initUI()

        # hyper parameter
        self.init_dir = './'
        self.extensions = ['.avi', '.mp4']
        self.wait_rate = 1
        self.plot_limit = 150
        self.thresh = [180, 255]  # [min, max]
        self.ref_thresh = 255
        self.add_radius = 10
        ### 동공 영역 필터링 추가 hyper parameter
        self.area_condition = 100
        self.symmetry_condition = 3
        self.fill_condition = 4

        # 변수 초기화 : PyQt
        self.video_paths = []
        self.display_video = ''
        self.change_video = False
        self.clicked = False
        self.clicked_start = False
        self.clicked_save_csv = False
        self.press_esc = False
        self.timerStep = 0

        # 변수 초기화 : OpenCV
        self.cap = None
        self.display_img = False
        self.ori_img = None
        self.roi_coord = []  # [x1, y1, x2, y2]
        self.horizontalSlider_max.setValue(self.thresh[1])
        self.horizontalSlider_min.setValue(self.thresh[0])
        self.horizontalSlider_reflection_min.setValue(self.ref_thresh)
        self.label_reflection_min.setText(f'{self.ref_thresh}')
        self.label_maxThr.setText(f'{self.thresh[1]}')  # 슬라이더로 설정한 max임계치값 보이기
        self.label_minThr.setText(f'{self.thresh[0]}')  # 슬라이더로 설정한 min임계치값 보이기
        self.pupil_info = []
        self.change_frame = False
        self.total_frames = 0
        ### 동공영역 필터링 변수 초기화
        self.horizontalSlider_area.setValue(self.area_condition)
        self.horizontalSlider_symmetry.setValue(self.symmetry_condition)
        self.horizontalSlider_fillcond.setValue(self.fill_condition)
        ### 동공영역 필터링 Slider 라벨
        self.label_area.setText(f'{self.area_condition}')
        self.label_symmetry.setText(f'{self.symmetry_condition}')
        self.label_fillcond.setText(f'{self.fill_condition}')


        # 동공 크기 csv 저장 변수
        self.csv_file = None
        self.plot_xs = []
        # self.plot_ys_radius = []
        self.plot_ys_diameter = []

        # 버튼에 기능 연결 (객체 탐색기)
        self.pushButton_GetFiles.clicked.connect(self.getFilesButton)  # 비디오 선택
        # self.pushButton_start.clicked.connect(self.startMeasurement)
        self.pushButton_saveDirectory.clicked.connect(self.selectDirectory_button)  # csv 저장폴더 선택
        self.pushButton_csvSave.clicked.connect(self.save_csv)  # csv파일 저장
        self.pushButton_quit.clicked.connect(self.program_quit)  # 프로그램 종료

        self.listWidget_video.itemDoubleClicked.connect(self.selectVideo)  # 선택한 비디오를 리스트로 보여주는 박스
        self.horizontalSlider_max.valueChanged.connect(self.maxThresh)  # 동공분할 max threshold 설정
        self.horizontalSlider_min.valueChanged.connect(self.minThresh)  # 동공분할 min threshold 설정
        self.horizontalSlider_reflection_min.valueChanged.connect(self.refThresh)  # 반사광분할 min threshold 설정
        self.horizontalSlider_video.sliderMoved.connect(self.video_frame)  # 비디오 timebar
        self.horizontalSlider_video.valueChanged.connect(self.video_frame_keyboard)
        ### 동공영역 필터링 Slider 버튼기능연결(경원오빠 추가)
        self.horizontalSlider_area.valueChanged.connect(self.area_Condition)
        self.horizontalSlider_symmetry.valueChanged.connect(self.symmetry_Condition)
        self.horizontalSlider_fillcond.valueChanged.connect(self.fillCond)

    # 사용자가 설정한 저장경로+파일명으로 csv파일에 write해 저장하기 
    def save_csv(self):
        self.clicked_save_csv = True

        csv_saveDir = self.label_saveDirectory.text()  # csv파일을 저장할 경로
        save_name = self.plainTextEdit_csvName.toPlainText()  # 저장할 csv 파일명
        self.csv_file = open(f'{csv_saveDir}/{save_name}.csv', 'w', newline='')
        csvwriter = csv.writer(self.csv_file)  # csv파일에 write
        # csvwriter.writerow(['frame', 'pupil_size_radius', 'pupil_size_diameter'])
        # for xs, ys_radius, ys_diameter in zip(self.plot_xs, self.plot_ys_radius, self.plot_ys_diameter):
        #     csvwriter.writerow([f'{xs}', f'{ys_radius}', f'{ys_diameter}'])
        csvwriter.writerow(['frame', 'pupil_size_diameter'])
        for xs, ys_diameter in zip(self.plot_xs, self.plot_ys_diameter):
            csvwriter.writerow([f'{xs}', f'{ys_diameter}'])
            
        self.csv_file.close()

    # 뭔지 잘 모르겠음
    # 비디오의 timebar가 변하지 않을 때만 동공크기를 검출하도록 제한?
    def video_frame_keyboard(self):
        if not self.change_frame:
            pass
        else:
            self.video_frame()

    # 비디오의 시간과 프레임 표시, 시간별 동공크기 그래프 표시
    def video_frame(self):
        self.change_frame = True

        frame_idx = self.horizontalSlider_video.value() # 비디오 timebar에서 받아온 프레임 넘버
        if frame_idx >= self.total_frames - 1:  # timebar가 비디오 전체 길이보다 큰 경우에는 pass
            pass
        else:
            show_str = f'{frames_to_timecode(frame_idx)}/{frames_to_timecode(self.total_frames)}'
            self.label_videoTime.setText(show_str)  # 전체 비디오 길이 중 현재 시간 표시
            self.label_videoFrame.setText(f'{frame_idx}/{self.total_frames}') # 전체 프레임수 중 현재 프레임 표시

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.display_img, self.ori_img = self.cap.read()
            self._showImage(self.ori_img, self.display_label, True)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # 그래프 범위 설정
            s_idx = frame_idx - self.plot_limit
            s_idx = 0 if s_idx < 0 else s_idx
            s_idx = self.total_frames - self.plot_limit if s_idx > self.total_frames - self.plot_limit else s_idx
            e_idx = frame_idx + self.plot_limit
            e_idx = int(self.total_frames - 1) if e_idx > self.total_frames else e_idx

            # 그래프 그리기
            show_xs = self.plot_xs[s_idx: e_idx]
            show_ys_diameter = self.plot_ys_diameter[s_idx: e_idx]
            graph = self.getGraph(show_xs, show_ys_diameter, frame_idx)

            self._showImage(graph, self.display_graph) # 시간별 동공크기 그래프 표시

    #
    def startMeasurement(self):
        self.clicked_start = True
        self.change_video = False  # 비디오를 바꿨는지
        csv_saveDir = self.label_saveDirectory.text()
        if self.cap and csv_saveDir:  # 비디오의 프레임이 잘 들어오고 저장경로가 설정되면
            while True:
                # if self.press_esc or self.change_video or self.clicked:
                if self.press_esc or self.change_video or self.clicked_save_csv:
                    self.fig = plt.figure()
                    # self.plot_xs = []
                    # self.plot_ys = []
                    self.max_y = 0
                    self.roi_coord = []
                    self.pupil_info = []
                    self.clicked = False
                    self.clicked_save_csv = False
                    self.change_video = True
                    self.csv_file = None
                    break

                self.display_img, self.ori_img = self.cap.read()

                if self.display_img and not self.change_frame:
                    idx_of_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    x1, x2, y1, y2 = 0, 0, 0, 0

                    if self.roi_coord and not self.clicked:
                        x1, y1, x2, y2 = self.roi_coord
                        height, width, _ = self.ori_img.shape
                        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                        roi = self.ori_img[y1:y2, x1:x2].copy()
                    else:
                        roi = self.ori_img.copy()

                    # 반사광 제거
                    roi = fill_reflected_light(roi, self.ref_thresh)
                    if self.roi_coord and not self.clicked:
                        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                        self.ori_img[y1:y2, x1:x2] = roi
                    else:
                        self.ori_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    # 동공 정보 (위치, 크기)
                    self.pupil_info, binary_eye = getPupil(roi, self.thresh,
                                                           self.area_condition,
                                                           self.symmetry_condition,
                                                           self.fill_condition)

                    ### 동공 영역 필터링

                    # 동공 크기 값 측정
                    if self.pupil_info:
                        roi, max_val = get_pupil_size(roi, binary_eye, self.pupil_info, self.add_radius)

                        self.ori_img[y1:y2, x1:x2] = roi
                        self.plot_ys_diameter[idx_of_frame] = max_val
                        # self.plot_ys_radius[idx_of_frame] = self.pupil_info[0][1]
                    else:
                        self.plot_ys_diameter[idx_of_frame] = 0
                        # self.plot_ys_radius[idx_of_frame] = 0

                    if self.checkBox_showGraph.isChecked():
                        # sequence graph
                        # idx_of_frame`
                        s_idx = idx_of_frame - self.plot_limit
                        s_idx = 0 if s_idx < 0 else s_idx
                        s_idx = self.total_frames - self.plot_limit if s_idx > self.total_frames - self.plot_limit else s_idx
                        e_idx = idx_of_frame + self.plot_limit
                        e_idx = int(self.total_frames - 1) if e_idx > self.total_frames else e_idx

                        show_xs = self.plot_xs[s_idx: e_idx]
                        # show_ys_radius = self.plot_ys_radius[s_idx: e_idx]
                        show_ys_diameter = self.plot_ys_diameter[s_idx: e_idx]
                        graph = self.getGraph(show_xs, show_ys_diameter, idx_of_frame)

                        self._showImage(graph, self.display_graph)

                    show_str = f'{frames_to_timecode(idx_of_frame)}/{frames_to_timecode(self.total_frames)}'
                    self.label_videoTime.setText(show_str)
                    self.label_videoFrame.setText(f'{idx_of_frame}/{self.total_frames}')
                    self._showImage(self.ori_img, self.display_label)
                    self._showImage(binary_eye, self.display_binary)

                    self.horizontalSlider_video.setValue(idx_of_frame)
                    cv2.waitKey(self.wait_rate)
                elif self.display_img:
                    # self.save_csv()
                    break
                else:
                    self.save_csv()
                    break

        self.clicked_start = False

    # 사용자가 지정한 roi영역에 박스 그리기, 검출한 동공영역에 원 그리기
    def _showImage(self, img, display_label, slide=False):

        if display_label is self.display_binary:  # 메인 라벨인 경우
            draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 3차원만 그려져서 3차원으로 변환
        elif display_label is self.display_label and slide:
            draw_img = img.copy()
            height, width, _ = img.shape
            if self.roi_coord:  # 사용자가 지정한 roi영역에 빨간색 박스 그리기
                x1, y1, x2, y2 = self.roi_coord
                draw_img = cv2.rectangle(draw_img,
                                         (int(x1 * width), int(y1 * height)),
                                         (int(x2 * width), int(y2 * height)),
                                         (0, 0, 255), 2)
        elif display_label is self.display_label:
            draw_img = img.copy()
            height, width, _ = img.shape

            if self.roi_coord:  # 사용자가 지정한 roi영역에 빨간색 박스 그리기
                x1, y1, x2, y2 = self.roi_coord
                draw_img = cv2.rectangle(draw_img,
                                         (int(x1 * width), int(y1 * height)),
                                         (int(x2 * width), int(y2 * height)),
                                         (0, 0, 255), 2)
            if self.pupil_info:  # 검출한 동공 영역에 파란색 원 그리기
                for info in self.pupil_info[:1]:
                    x, y = info[0]
                    if self.roi_coord:
                        x, y = int(x + self.roi_coord[0] * width), int(y + self.roi_coord[1] * height)  # 그릴때 다시 절대좌표로 바꿈
                    cv2.circle(draw_img, (x, y), info[1] + self.add_radius, (255, 0, 0), 2)
        else:
            draw_img = img.copy()  # 그래프 그리려고 복사

        qpixmap = cvtPixmap(draw_img, (display_label.width(), display_label.height())) # 이미지를 읽어서 pyqt로 보여줌
        display_label.setPixmap(qpixmap)

    # 잘 모르겠음
    def selectVideo(self):
        self.idx_video = self.listWidget_video.currentRow()  # 현재 셀의 행을 반환
        # video_paths안에는 getFilesButton()에서 받아온 사용자가 선택한 비디오 파일의 경로가 담겨있음
        self.cap = cv2.VideoCapture(self.video_paths[self.idx_video])  # 해당 경로의 비디오를 불러옴
        self.display_img, self.ori_img = self.cap.read()  # 비디오를 한 frame씩 읽어옴
        # frame을 제대로 읽으면 display_img값이 True, 실패하면 False
        # ori_img에는 읽어온 frame값이 들어있음
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 화면 크기 설정, cv2.CAP_PROP_POS_FRAMES는 현재 프레임 개수
        if self.display_img: # 영상의 프레임을 제대로 읽어오면 사용하는 변수 초기화
            self.fig = plt.figure()
            self.max_y = 0
            self.roi_coord = []
            self.pupil_info = []
            self.clicked = False
            self.change_video = True
            self.csv_file = None
            self.change_frame = True
            self.clicked_save_csv = False
            avi_filename = os.path.split(self.video_paths[self.idx_video])
            filename = os.path.splitext(avi_filename[-1])[0]
            self.plainTextEdit_csvName.setPlainText(filename)
            self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.plot_xs = list(range(int(self.total_frames)))
            # self.plot_ys_radius = [0] * int(self.total_frames + 1)
            self.plot_ys_diameter = [0] * int(self.total_frames + 1)
            self.horizontalSlider_video.setRange(0, self.total_frames - 1)
            self._showImage(self.ori_img, self.display_label)  # 마지막에 이미지 뿌려주기, 어디에 뿌릴지 객체를 넣어줌

    # 클릭했을 때
    def mousePressEvent(self, event):
        if self.display_img:
            # event.x()는 현재 마우스의 x좌표를 반환
            rel_x = (event.x() - self.display_label.x()) / self.display_label.width()  # 상대좌표로 바꾸기
            rel_y = (event.y() - self.display_label.y()) / self.display_label.height()

            if 0 <= rel_x <= 1 and 0 <= rel_y < 1:
                self.clicked = True
                self.roi_coord = [rel_x, rel_y, rel_x, rel_y]

            self._showImage(self.ori_img, self.display_label, slide=True)

    # 클릭하고 움직였을 때
    def mouseMoveEvent(self, event):
        if self.display_img and self.clicked:
            rel_x = (event.x() - self.display_label.x()) / self.display_label.width()
            rel_y = (event.y() - self.display_label.y()) / self.display_label.height()

            if 0 <= rel_x <= 1 and 0 <= rel_y < 1:
                self.roi_coord[2] = rel_x
                self.roi_coord[3] = rel_y
            elif rel_x > 1:
                self.roi_coord[2] = 1
            elif rel_y > 1:
                self.roi_coord[3] = 1
            elif rel_x < 0:
                self.roi_coord[2] = 0
            elif rel_y < 0:
                self.roi_coord[3] = 0

            self._showImage(self.ori_img, self.display_label, slide=True)

    # 마우스 뗐을 떄
    def mouseReleaseEvent(self, event):
        if  self.clicked:
            self.clicked = False  # 이게 포인트

            x1, y1, x2, y2 = self.roi_coord
            if x1 == x2 or y1 == y2:  # 클릭만하고 roi를 설정하지 않았을 떄
                self.roi_coord = []
            else:  # roi를 설정할 때 x1, y1이 x2, y2보다 작은 좌표를 가지도록
                if x1 > x2:
                    self.roi_coord[2] = x1
                    self.roi_coord[0] = x2
                if y1 > y2:
                    self.roi_coord[3] = y1
                    self.roi_coord[1] = y2

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.press_esc = True
            self.close()
        elif e.key() == Qt.Key_Space:
            self.clicked_start = True
            if not self.change_frame:  # 프레임변화가 없을 때만 동공검출을 하도록
                self.change_frame = True 
            else:
                self.change_frame = False
                self.startMeasurement()

    # 사용자가 선택한 csv저장폴더의 경로를 반환
    def selectDirectory_button(self):
        self.saved_dir = QFileDialog.getExistingDirectory(self, 'Select save directory', './')
        self.label_saveDirectory.setText(self.saved_dir)  # 빈 박스에 선택한 저장경로 담기

    # 사용자가 선택한 비디오 파일을 반환
    def getFilesButton(self):
        self.video_paths = [] # 비디오 경로
        self.video_paths = QFileDialog.getOpenFileNames(self, 'Get Files', self.init_dir)[0] # 사용자가 선택한 파일을 반환
        if self.video_paths:
            temp = []  # 비디오 확장자 담기(확장자 2가지임)
            for i, path in enumerate(self.video_paths):
                if os.path.splitext(path)[-1] in self.extensions:
                    temp.append(path)
            self.video_paths = temp

            for i, path in enumerate(self.video_paths):
                self.listWidget_video.insertItem(i, os.path.basename(path))  # i위치에 선택한 비디오 보여주는 박스에 넣기
        else:
            self.listWidget_video.clear()  # 사용자가 선택한 파일이 없으면 빈칸

    # 반사광분할에 사용되는 min threshold값을 설정, 동공검출 결과 보여주기
    def refThresh(self):
        x1, y1, x2, y2 = 0, 0, 0, 0

        self.ref_thresh = self.horizontalSlider_reflection_min.value()
        self.label_reflection_min.setText(f'{self.ref_thresh}') # 설정한 min threshold값 GUI에 보여주기

        if self.cap and not self.clicked_start:
            post_img = self.ori_img.copy()

            if self.roi_coord and not self.clicked:
                x1, y1, x2, y2 = self.roi_coord
                height, width, _ = self.ori_img.shape
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                roi = self.ori_img[y1:y2, x1:x2].copy()
            else:
                roi = self.ori_img.copy()

            roi = fill_reflected_light(roi, self.ref_thresh)
            if self.roi_coord:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                post_img[y1:y2, x1:x2] = roi
            else:
                post_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            self.pupil_info, binary_eye = getPupil(roi, self.thresh,
                                                   self.area_condition,
                                                   self.symmetry_condition,
                                                   self.fill_condition)

            self._showImage(post_img, self.display_label)
            self._showImage(binary_eye, self.display_binary)

    # 동공 후보영역 조건1 - 최소원의 크기
    def area_Condition(self):
        x1, y1, x2, y2 = 0, 0, 0, 0

        self.area_condition = self.horizontalSlider_area.value()
        self.label_area.setText(f'{self.area_condition}')  # 설정한 값을 GUI에 표시

        if self.cap and not self.clicked_start:
            post_img = self.ori_img.copy()

            if self.roi_coord and not self.clicked:
                x1, y1, x2, y2 = self.roi_coord
                height, width, _ = self.ori_img.shape
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                roi = self.ori_img[y1:y2, x1:x2].copy()
            else:
                roi = self.ori_img.copy()

            roi = fill_reflected_light(roi, self.ref_thresh)
            if self.roi_coord:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                post_img[y1:y2, x1:x2] = roi
            else:
                post_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            self.pupil_info, binary_eye = getPupil(roi, self.thresh,
                                                   self.area_condition,
                                                   self.symmetry_condition,
                                                   self.fill_condition)

            self._showImage(post_img, self.display_label)
            self._showImage(binary_eye, self.display_binary)

    # 동공 후보영역 조건2 - 종횡비
    def symmetry_Condition(self):
        x1, y1, x2, y2 = 0, 0, 0, 0

        self.symmetry_condition = self.horizontalSlider_symmetry.value()
        self.label_symmetry.setText(f'{self.symmetry_condition}')


        if self.cap and not self.clicked_start:
            post_img = self.ori_img.copy()

            if self.roi_coord and not self.clicked:
                x1, y1, x2, y2 = self.roi_coord
                height, width, _ = self.ori_img.shape
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                roi = self.ori_img[y1:y2, x1:x2].copy()
            else:
                roi = self.ori_img.copy()

            roi = fill_reflected_light(roi, self.ref_thresh)
            if self.roi_coord:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                post_img[y1:y2, x1:x2] = roi
            else:
                post_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            self.pupil_info, binary_eye = getPupil(roi, self.thresh,
                                                   self.area_condition,
                                                   self.symmetry_condition,
                                                   self.fill_condition)

            self._showImage(post_img, self.display_label)
            self._showImage(binary_eye, self.display_binary)

    # 동공 후보영역 조건3 - 원이 비어있는 비율
    def fillCond(self):
        x1, y1, x2, y2 = 0, 0, 0, 0

        self.fill_condition = self.horizontalSlider_fillcond.value()
        self.label_fillcond.setText(f'{self.fill_condition}')

        if self.cap and not self.clicked_start:
            post_img = self.ori_img.copy()

            if self.roi_coord and not self.clicked:
                x1, y1, x2, y2 = self.roi_coord
                height, width, _ = self.ori_img.shape
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                roi = self.ori_img[y1:y2, x1:x2].copy()
            else:
                roi = self.ori_img.copy()

            roi = fill_reflected_light(roi, self.ref_thresh)
            if self.roi_coord:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                post_img[y1:y2, x1:x2] = roi
            else:
                post_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            self.pupil_info, binary_eye = getPupil(roi, self.thresh,
                                                   self.area_condition,
                                                   self.symmetry_condition,
                                                   self.fill_condition)

            self._showImage(post_img, self.display_label)
            self._showImage(binary_eye, self.display_binary)

    # 동공분할에 사용되는 max threshold값을 설정, 동공검출 결과 보여주기
    # 이 경우에는 max threshold는 255로 고정됨
    def maxThresh(self):
        x1, y1, x2, y2 = 0, 0, 0, 0

        self.thresh[1] = self.horizontalSlider_max.value()  # GUI 버튼에서 받아온 값
        if self.thresh[1] <= self.thresh[0]:  # max 임계치가 min 임계치보다는 큰 값을 가지도록 
            self.thresh[1] = self.thresh[0] + 1
        self.horizontalSlider_max.setValue(self.thresh[1])  # 조정한 값으로 max 임계치 설정
        self.label_maxThr.setText(f'{self.thresh[1]}')  # 설정한 max임계치값 GUI에 보여주기

        if self.cap and not self.clicked_start:
            post_img = self.ori_img.copy()

            if self.roi_coord and not self.clicked:
                x1, y1, x2, y2 = self.roi_coord
                height, width, _ = self.ori_img.shape
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                roi = self.ori_img[y1:y2, x1:x2].copy()
            else:
                roi = self.ori_img.copy()

            roi = fill_reflected_light(roi, self.ref_thresh) # 반사광 제거
            if self.roi_coord:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # 컬러로 변환해서 시각화
                post_img[y1:y2, x1:x2] = roi
            else:
                post_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            self.pupil_info, binary_eye = getPupil(roi, self.thresh,
                                                   self.area_condition,
                                                   self.symmetry_condition,
                                                   self.fill_condition)

            self._showImage(post_img, self.display_label)  # 동공 검출 결과 보여주기
            self._showImage(binary_eye, self.display_binary)  # 이진화된 이미지 보여주기

    # 동공분할에 사용되는 min threshold값을 설정, 동공검출 결과 보여주기
    def minThresh(self):
        x1, y1, x2, y2 = 0, 0, 0, 0

        self.thresh[0] = self.horizontalSlider_min.value()
        if self.thresh[0] >= self.thresh[1]:
            self.thresh[0] = self.thresh[1] - 1
        self.horizontalSlider_min.setValue(self.thresh[0])
        self.label_minThr.setText(f'{self.thresh[0]}')

        if self.cap and not self.clicked_start:
            post_img = self.ori_img.copy()

            if self.roi_coord and not self.clicked:
                x1, y1, x2, y2 = self.roi_coord
                height, width, _ = self.ori_img.shape
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)  # 상대좌표 구하기
                roi = self.ori_img[y1:y2, x1:x2].copy()
            else:
                roi = self.ori_img.copy()

            roi = fill_reflected_light(roi, self.ref_thresh)  # 반사광 채우기
            if self.roi_coord:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                post_img[y1:y2, x1:x2] = roi
            else:
                post_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            self.pupil_info, binary_eye = getPupil(roi, self.thresh,
                                                   self.area_condition,
                                                   self.symmetry_condition,
                                                   self.fill_condition)

            self._showImage(post_img, self.display_label)
            self._showImage(binary_eye, self.display_binary)

    # 시간별 동공크기 그래프 그리기
    def getGraph(self, xs, ys_diameter, pre_idx):
        max_y = max(ys_diameter)
        if self.max_y < max_y:
            self.max_y = max_y

        ax = self.fig.add_subplot(1, 1, 1)

        ax.clear()
        # ax.plot(xs, ys_radius)
        ax.plot(xs, ys_diameter)

        plt.scatter(pre_idx, max_y + 8, marker='o', color='salmon')
        plt.ylim(-1, self.max_y + 10)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Pupil size')

        # 위에서 구한 시간별 동공크기를 넘파이로 바꾸기, 나중에 다시 cvtPixmap으로 바꿔야함
        self.fig.canvas.draw()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    # 화면의 중앙 좌표를 계산해 윈도우 중앙에 UI창을 사각형으로 생성하기
    def _center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # UI창 이름, 전체 구조 띄우기
    def initUI(self):
        self.setWindowTitle('Visual fatigue measurement')
        self.setWindowIcon(QIcon('icon.jpg'))
        self.setFixedSize(1198, 746)
        self._center()
        self.show()

    # 프로그램 종료
    def program_quit(self):
        self.press_esc = True
        QCoreApplication.instance().quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
