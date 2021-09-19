# visual-fatigue-analysis
> 본 프로젝트는 동공의 움직임을 기반으로 시각 피로도를 정량적으로 측정합니다. 
눈 깜빡임 빈도, 눈 감은 시간, 동공 변화 속도 3가지를 피로도 측정 지표로 사용했습니다.


## Description

## Environment
Windows 10

## Prerequisite
python 3.8.5
numpy 1.19.2
matplotlib 3.2.2
scipy 1.5.3
pandas 1.2.0


```python
(git bash)
git clone https://github.com/HanNayeoniee/visual-fatigue-analysis.git
pip install numpy
pip install matplotlib
pip install scipy
pip install pandas
```


## Files
1. csv_preprocess.py : csv파일 보간

```interpolation_zero()``` : 앞/뒤 프레임 사이에 동공크기가 0인 프레임이 하나일때 앞/뒤 프레임의 평균으로 대체
```interpolation_nonzero()``` : 앞/뒤 프레임 값이 모두 0일때 가운데 프레임 값을 0으로 대체
```thres_zero()``` : 동공 크기가 평균의 1/3이하인 값은 모두 0으로 대체

2. blink.py : 눈 깜빡임 빈도, 눈 감은 시간 계산

```count_blink()``` : 눈 깜빡임 빈도, 눈 감은 시간 계산
```draw_graph()``` : 그래프 그리기
```draw_trendline_blink()``` : 2차식 추세선과 함께 그래프 그리기

3. fft.py : 동공 변화 속도 계산 

```del_high_freq()``` : 고주파 필터링
```del_high_and_low_freq()``` : 고주파, 저주파 필터링
```fft()``` : 동공 변화 속도 계산
```draw_fft_graph()``` : 그래프 그리기
```draw_trendline_fft()``` : 2차식 추세선과 함께 그래프 그리기

4. main.py :
전처리 -> 눈 깜빡임 빈도, 눈 감은 시간 계산 -> 동공크기 변화율 계산 ->엑셀 파일, 결과 그래프 저장

## Usage



## Paper
ICICT 2021
[Visual fatigue analysis of welding mask wearer based on blink analysis](https://drive.google.com/file/d/1VjO1nBAddad340xkDOA79pc2VNdl815T/view?usp=sharing)



## Authors

|                 한나연               |                 진경원                |              목지원               |
| :------------------------------------------: | :-----------------------------------------: | :----------------------------------------: |
| <img src="https://user-images.githubusercontent.com/33839093/129561824-7f779bf8-8036-4ab6-812e-4c7aa12c3d79.png" width=150px> | <img src="https://user-images.githubusercontent.com/33839093/129561437-e778deff-86fd-4f7e-b938-38a1125578ea.png" width=150px> | <img src="https://user-images.githubusercontent.com/33839093/129562599-c27f52c9-31cc-4f25-916e-ce0dbee6f315.jpg" width=150px> |
|                   **[Github](https://github.com/HanNayeoniee)**                   |                   **[Github](https://github.com/KyungwonJIN)**                   |               **[Github](https://github.com/mjw2705)**               |


