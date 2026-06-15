# AI 기반 실시간 졸음 및 집중도 감지 시스템

> 하품 횟수, 눈을 감은 이미지와 눈을 뜬 이미지, 운전자의 다양한 상태 데이터를 활용한 졸음 예측 AI 프로젝트 입니다.

## 1. 프로젝트 개요

본 프로젝트는 웹캠 영상을 기반으로 사용자의 졸음 및 집중도 저하 상태를 실시간으로 감지하는 컴퓨터비전 프로젝트이다.

단일 이미지 한 장만으로 졸음 여부를 판단하는 방식은 실제 서비스에서 한계가 있기 때문에, 본 프로젝트에서는 웹캠 프레임을 일정 간격으로 분석하고 최근 30~60초 동안의 프레임 기록을 누적하여 최종 상태를 판단한다.

주요 기능은 다음과 같다.

* 웹캠 기반 실시간 졸음 감지
* CNN 모델을 이용한 `normal / drowsy` 이진 분류
* MediaPipe Face Mesh 기반 눈 감김, 입 벌림, 얼굴 방향 분석
* 최근 30초 기준 PERCLOS 계산
* 눈 감김 지속 시간 측정
* 최근 60초 기준 하품 횟수 측정
* 정상, 졸음 의심, 졸음 감지 상태 표시
* 모델별 성능 비교

---

## 2. 사용 기술

| 구분                  | 기술          | 사용 이유                                       |
| ------------------- | ----------- | ------------------------------------------- |
| Python              | Python      | 전체 데이터 처리, 학습, 서버 구현                        |
| PyTorch             | torch       | CNN 모델 학습 및 추론                              |
| Torchvision         | torchvision | 이미지 전처리, 데이터 로딩, pretrained vision model 사용 |
| OpenCV              | cv2         | 영상 데이터에서 프레임 추출                             |
| PIL/Pillow          | PIL         | 이미지 열기, RGB 변환, resize, 저장                  |
| MediaPipe           | Face Mesh   | 눈, 입, 얼굴 방향 분석을 위한 얼굴 랜드마크 추출               |
| Flask               | Flask       | 웹캠 프레임을 서버로 받아 분석하는 API 구현                  |
| HTML/CSS/JavaScript | Frontend    | 웹캠 실행 및 분석 결과 표시                            |
| AWS EC2             | Cloud       | 모델 학습 및 서버 실행 환경                            |

### Torchvision 설명

Torchvision은 PyTorch에서 컴퓨터비전 작업을 쉽게 수행하기 위한 라이브러리이다. 본 프로젝트에서는 MobileNetV3, ResNet18, EfficientNet-B0 모델을 불러오고, 이미지 resize, normalization, augmentation, ImageFolder 데이터 로딩에 사용하였다.

### OpenCV 설명

OpenCV는 이미지와 영상을 처리하기 위한 대표적인 컴퓨터비전 라이브러리이다. 본 프로젝트에서는 YawDD와 같은 영상 데이터에서 일정 간격으로 프레임을 추출하기 위해 사용하였다. 영상 파일을 열고, 특정 프레임을 읽고, 이미지로 변환하는 과정이 컴퓨터비전 전처리에 해당하므로 OpenCV를 Computer Vision 기술로 분류하였다.

### PIL/Pillow 설명

PIL은 Python Imaging Library의 약자이며, 현재는 Pillow 라이브러리로 사용된다. 본 프로젝트에서는 이미지 파일을 열고 RGB 형식으로 변환한 뒤, 224×224 크기로 조정하고 JPG 파일로 저장하는 데 사용하였다.

---

## 3. 사용 데이터셋

본 프로젝트에서는 Kaggle에서 제공되는 졸음, 눈 감김, 하품 관련 데이터셋을 사용하였다.

| 데이터셋                   | 데이터 형태    | 사용 목적                 |
| ---------------------- | --------- | --------------------- |
| MRL Eye Dataset        | 눈 이미지     | 눈 뜸/눈 감김 상태 학습        |
| Eye Open Close Dataset | 눈 이미지     | 눈 상태 데이터 보강           |
| YawDD Dataset          | 하품/비하품 영상 | 하품 상태 데이터 확보 및 프레임 추출 |

최종 라벨은 다음과 같이 이진 분류로 통합하였다.

| 원본 상태                            | 최종 라벨  |
| -------------------------------- | ------ |
| open eye, no_yawn, normal, awake | normal |
| closed eye, yawn, sleepy, drowsy | drowsy |

즉, 눈을 뜨고 있거나 하품하지 않는 상태는 `normal`, 눈을 감고 있거나 하품하는 상태는 `drowsy`로 분류하였다.

---

## 4. 데이터 병합 및 라벨링 방식

여러 데이터셋은 `data/raw` 폴더 아래에 저장한 뒤, 전처리 코드가 전체 파일을 재귀적으로 탐색하였다.

처리 대상 파일 확장자는 다음과 같다.

* 이미지: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
* 영상: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

라벨은 파일명과 폴더명에 포함된 키워드를 기준으로 자동 추론하였다.

normal 라벨 키워드:

```text
no_yawn, no-yawn, noyawn, no yawn, not_yawn, normal, awake, open, open_eye, open-eyes, openeyes, opened
```

drowsy 라벨 키워드:

```text
closed, close, closed_eye, closed-eyes, closedeyes, sleep, sleepy, drowsy, yawn, yawning
```

전처리 코드에서는 normal 키워드를 먼저 검사하고, 그 다음 drowsy 키워드를 검사하였다.
이는 `no_yawn`이라는 단어 안에 `yawn`이 포함되어 있기 때문에, drowsy 키워드를 먼저 검사하면 `no_yawn` 데이터가 잘못해서 drowsy로 분류될 수 있기 때문이다.

라벨 키워드가 없는 파일은 normal/drowsy로 확실하게 분류할 수 없기 때문에 제외하였다.

---

## 5. 영상 데이터 처리 방식

YawDD와 같은 영상 데이터는 영상 전체를 직접 학습하지 않고, 일정 간격으로 프레임을 추출하여 이미지 데이터로 변환하였다.

전처리 설정은 다음과 같다.

```bash
python src/prepare_dataset_fast.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --image_size 224 \
  --video_every_n_frames 300 \
  --max_video_frames_per_video 20 \
  --max_per_class 3000
```

영상 처리 과정은 다음과 같다.

```text
1. OpenCV의 cv2.VideoCapture로 영상 파일을 연다.
2. 전체 프레임 수를 확인한다.
3. 0, 300, 600, 900 ... 번째 프레임을 선택한다.
4. 영상 하나당 최대 20장까지만 추출한다.
5. OpenCV의 BGR 이미지를 RGB 이미지로 변환한다.
6. 224×224 크기로 resize한다.
7. 임시 폴더에 JPG 이미지로 저장한다.
8. 이후 이미지 데이터와 동일하게 train/val/test 데이터셋에 병합한다.
```

이렇게 한 이유는 영상의 모든 프레임을 사용하면 데이터 수가 너무 많아지고, 비슷한 프레임이 과도하게 반복되어 학습 효율이 떨어질 수 있기 때문이다.

---

## 6. 전처리에서 제외한 데이터

전처리 과정에서 다음 데이터는 제외하였다.

| 제외 대상                 | 제외 이유                         |
| --------------------- | ----------------------------- |
| 라벨 키워드가 없는 파일         | normal/drowsy로 라벨링할 수 없음      |
| 이미지/영상 확장자가 아닌 파일     | 학습에 사용하지 않는 파일 형식             |
| 깨진 이미지 또는 열 수 없는 이미지  | PIL에서 로딩 실패                   |
| 영상의 모든 프레임            | 중복이 많고 데이터가 과도하게 증가함          |
| 영상 하나당 20장을 초과하는 프레임  | 특정 영상이 데이터셋을 과도하게 차지하지 않도록 제한 |
| 클래스당 3,000장을 초과하는 데이터 | 클래스 균형 유지 및 학습 시간 단축          |

---

## 7. 최종 데이터 구성

클래스 불균형을 줄이기 위해 `normal`과 `drowsy`의 개수를 동일하게 맞춘 균형 데이터셋을 구성하였다.

본 프로젝트에서는 클래스당 최대 3,000장을 사용하였다.

| Label  | Count |
| ------ | ----: |
| normal | 3,000 |
| drowsy | 3,000 |
| total  | 6,000 |

데이터는 train/validation/test로 70:15:15 비율로 나누었다.
또한 `stratify` 옵션을 사용하여 각 split에서 normal과 drowsy 비율이 동일하게 유지되도록 하였다.

| Split      | normal | drowsy | total |
| ---------- | -----: | -----: | ----: |
| train      |  2,100 |  2,100 | 4,200 |
| validation |    450 |    450 |   900 |
| test       |    450 |    450 |   900 |
| total      |  3,000 |  3,000 | 6,000 |

최종 데이터셋 구조는 다음과 같다.

```text
data/processed/
├── train/
│   ├── drowsy/
│   └── normal/
├── val/
│   ├── drowsy/
│   └── normal/
├── test/
│   ├── drowsy/
│   └── normal/
└── metadata.csv
```

`metadata.csv`에는 각 이미지의 split, label, 저장 경로, 원본 경로가 기록된다.

---

## 8. 이미지 전처리 및 증강

이미지에는 다음 공통 전처리를 적용하였다.

```text
1. 이미지 열기
2. RGB 형식으로 변환
3. 224×224 크기로 resize
4. Tensor 변환
5. ImageNet mean/std 기준 normalization
```

학습 데이터에는 일반화 성능을 높이기 위해 데이터 증강을 적용하였다.

* RandomHorizontalFlip
* RandomRotation
* ColorJitter

Validation과 Test 데이터에는 증강을 적용하지 않고 resize와 normalization만 적용하였다.
이는 검증 및 테스트 데이터는 실제 성능 평가용이므로 원본 분포를 유지해야 하기 때문이다.

---

## 9. 사용 모델

본 프로젝트에서는 3개의 CNN 모델을 비교하였다.

| 모델                | 특징                    |
| ----------------- | --------------------- |
| MobileNetV3-Small | 가볍고 빠른 모델로 실시간 추론에 유리 |
| ResNet18          | 안정적인 baseline 모델      |
| EfficientNet-B0   | 정확도와 연산량의 균형이 좋은 모델   |

모든 모델은 ImageNet pretrained weight를 기반으로 사용하였으며, 마지막 classifier layer를 `normal / drowsy` 이진 분류에 맞게 수정하였다.

---

## 10. 학습 및 평가 방식

학습에는 PyTorch를 사용하였다.

공통 학습 설정은 다음과 같다.

| 항목            | 내용                  |
| ------------- | ------------------- |
| 입력 크기         | 224×224 RGB         |
| 출력 클래스        | normal, drowsy      |
| Loss Function | CrossEntropyLoss    |
| Optimizer     | AdamW               |
| Scheduler     | ReduceLROnPlateau   |
| Best Model 기준 | Validation Accuracy |
| 평가 데이터        | Test Set            |

평가 지표는 다음과 같다.

| 지표               | 의미                       |
| ---------------- | ------------------------ |
| Accuracy         | 전체 샘플 중 정확히 예측한 비율       |
| Precision Macro  | 각 클래스 precision의 평균      |
| Recall Macro     | 각 클래스 recall의 평균         |
| F1 Macro         | precision과 recall의 조화 평균 |
| Confusion Matrix | 실제 라벨과 예측 라벨의 관계         |
| FPS              | 초당 처리 가능한 이미지 수          |
| Inference Time   | 이미지 1장 처리에 걸리는 평균 시간     |

---

## 11. PERCLOS 설명

PERCLOS는 `Percentage of Eye Closure`의 약자로, 최근 일정 시간 동안 눈이 감겨 있었던 비율을 의미한다.

본 프로젝트에서는 최근 30초 동안 분석된 프레임 중 눈이 감긴 프레임의 비율로 계산하였다.

```text
PERCLOS = 최근 30초 동안 눈 감김 프레임 수 / 최근 30초 전체 프레임 수
```

예를 들어 최근 30초 동안 60개의 프레임을 분석했고, 그중 21개 프레임에서 눈이 감겨 있었다면 다음과 같다.

```text
PERCLOS = 21 / 60 = 0.35 = 35%
```

PERCLOS는 단일 프레임이 아니라 일정 시간 동안의 눈 감김 비율을 보기 때문에 졸음 판단에 적합하다.

---

## 12. 실시간 상태 판단 기준

본 프로젝트의 실시간 서비스는 단일 프레임만으로 졸음 여부를 결정하지 않는다.
각 프레임에서 CNN 모델의 졸음 확률, 눈 감김 여부, 입 벌림 여부, 얼굴 방향 이탈 여부를 계산하고, 최근 30~60초 동안의 결과를 누적하여 최종 상태를 결정한다.

### 12.1 눈 감김 기준

MediaPipe Face Mesh 랜드마크를 이용해 눈의 세로/가로 비율인 EAR을 계산한다.

```text
EAR = 눈 세로 거리 / 눈 가로 거리
```

현재 기준은 다음과 같다.

```text
EAR < 0.21 → 눈 감김
```

| 조건                    | 상태    |
| --------------------- | ----- |
| 눈 감김 2초 이상 지속         | 졸음 감지 |
| 최근 30초 PERCLOS 25% 이상 | 졸음 의심 |
| 최근 30초 PERCLOS 35% 이상 | 졸음 감지 |

### 12.2 하품 기준

입의 세로/가로 비율인 MAR을 계산한다.

```text
MAR = 입 세로 거리 / 입 가로 거리
```

현재 기준은 다음과 같다.

```text
MAR > 0.32 → 입 벌림
입 벌림 상태가 1초 이상 지속 → 하품 1회
```

입이 계속 벌어진 상태에서는 중복 카운트하지 않는다.
입이 닫힌 뒤 다시 1초 이상 벌어졌을 때 새로운 하품으로 카운트한다.

| 조건              | 상태    |
| --------------- | ----- |
| 최근 60초 하품 2회 이상 | 졸음 의심 |
| 최근 60초 하품 3회 이상 | 졸음 감지 |

### 12.3 얼굴 방향 기준

얼굴 중심과 코 위치의 차이를 이용해 정면을 보고 있는지 판단하였다.

| 조건             | 상태              |
| -------------- | --------------- |
| 정면 이탈 5초 이상 지속 | 졸음 감지 또는 집중도 저하 |

### 12.4 CNN 평균 졸음 확률 기준

CNN 모델은 각 프레임에 대해 `normal`과 `drowsy` 확률을 출력한다.
이 값은 단일 프레임만으로 최종 판단하지 않고, 최근 30초 평균값으로 사용하였다.

| 조건                                      | 상태    |
| --------------------------------------- | ----- |
| 최근 30초 평균 졸음 확률 60% 이상                  | 졸음 의심 |
| 최근 30초 평균 졸음 확률 70% 이상 + PERCLOS 25% 이상 | 졸음 감지 |

---

## 13. 최종 상태 결정 기준

최종 상태는 세 단계로 구분하였다.

| 상태      | 의미    | 기준                                                                                         |
| ------- | ----- | ------------------------------------------------------------------------------------------ |
| normal  | 정상    | 위험 조건 없음                                                                                   |
| warning | 졸음 의심 | 평균 졸음 확률 60% 이상, PERCLOS 25% 이상, 최근 60초 하품 2회 이상                                           |
| drowsy  | 졸음 감지 | 눈 감김 2초 이상, PERCLOS 35% 이상, 최근 60초 하품 3회 이상, 얼굴 이탈 5초 이상, 평균 졸음 확률 70% 이상 + PERCLOS 25% 이상 |

즉, CNN 모델의 단일 프레임 예측은 보조 지표로 사용하고, 최종 알림은 눈 감김 지속 시간, PERCLOS, 하품 횟수, 얼굴 방향 이탈 시간과 같은 시간 기반 지표를 함께 고려하여 결정하였다.

---

## 14. 성능 결과 정리 방법

모델별 성능 결과는 각 output 폴더의 `metrics.json`에 저장된다.

```text
outputs_mobilenet/metrics.json
outputs_resnet18/metrics.json
outputs_efficientnet/metrics.json
```

성능표는 다음 명령어로 확인할 수 있다.

```bash
python - <<'PY'
import json
from pathlib import Path

models = {
    "MobileNetV3-Small": "outputs_mobilenet/metrics.json",
    "ResNet18": "outputs_resnet18/metrics.json",
    "EfficientNet-B0": "outputs_efficientnet/metrics.json",
}

print("| Model | Accuracy | Precision | Recall | F1-score | FPS | Inference Time |")
print("|---|---:|---:|---:|---:|---:|---:|")

for name, path in models.items():
    p = Path(path)
    m = json.loads(p.read_text())
    print(
        f"| {name} | "
        f"{m.get('accuracy', 0):.4f} | "
        f"{m.get('precision_macro', 0):.4f} | "
        f"{m.get('recall_macro', 0):.4f} | "
        f"{m.get('f1_macro', 0):.4f} | "
        f"{m.get('fps', 0):.2f} | "
        f"{m.get('avg_inference_time_sec_per_image', 0):.4f}s |"
    )
PY
```


| Model | Accuracy | Precision | Recall | F1-score | FPS | Inference Time |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 0.9733 | 0.9737 | 0.9733 | 0.9733 | 3361.35 | 0.0003s |
| ResNet18 | 0.9744 | 0.9747 | 0.9744 | 0.9744 | 1310.76 | 0.0008s |
| EfficientNet-B0 | 0.9678 | 0.9678 | 0.9678 | 0.9678 | 883.69 | 0.0011s |

---

## 15. 결과 해석

MobileNetV3-Small은 모델 크기가 작아 추론 속도가 빠르고 FPS가 높게 나올 가능성이 높다. 따라서 실시간 웹캠 서비스에 적합하다. 다만 모델 용량이 작아 복잡한 특징을 학습하는 능력은 상대적으로 제한될 수 있다.

ResNet18은 구조가 단순하고 안정적인 baseline 모델로, 성능과 속도 측면에서 중간 수준의 결과를 보일 수 있다.

EfficientNet-B0는 정확도와 연산량의 균형이 좋은 모델이기 때문에 높은 accuracy와 F1-score를 기대할 수 있다. 다만 MobileNetV3-Small보다 추론 속도는 느릴 수 있다.

세 모델 모두 높은 성능이 나올 경우, 이는 다음과 같이 해석할 수 있다.

* `normal / drowsy` 이진 분류 문제로 단순화되어 클래스 차이가 비교적 명확함
* 눈 뜸, 눈 감김, 하품은 시각적 특징 차이가 큼
* ImageNet pretrained 모델을 사용하여 적은 데이터로도 좋은 특징 추출 가능
* 클래스 균형을 맞춰 학습하여 한쪽 클래스 편향을 줄임

반대로 실제 웹캠 환경에서는 Kaggle 데이터셋과 다른 조명, 각도, 해상도, 얼굴 크기, 안경 착용 여부 등에 따라 성능이 달라질 수 있다. 따라서 최종 서비스에서는 CNN의 단일 프레임 예측만 사용하지 않고, EAR, MAR, PERCLOS, 하품 횟수, 얼굴 방향 이탈 시간과 같은 시간 기반 지표를 함께 사용하였다.

---

## 16. 실행 방법

### 16.1 학습

```bash
python src/train.py \
  --data_dir data/processed \
  --output_dir outputs_mobilenet \
  --model_name mobilenet_v3_small \
  --num_workers 2

python src/train.py \
  --data_dir data/processed \
  --output_dir outputs_resnet18 \
  --model_name resnet18 \
  --num_workers 2

python src/train.py \
  --data_dir data/processed \
  --output_dir outputs_efficientnet \
  --model_name efficientnet_b0 \
  --num_workers 2
```

### 16.2 웹캠 서비스 실행

```bash
export MODEL_PATH=outputs_efficientnet/best_model.pt
python app.py
```

브라우저에서 접속한다.

```text
http://localhost:8080
```

웹캠 시작 버튼을 누르면 프레임 분석이 시작되고, 화면에 다음 값들이 표시된다.

* 현재 상태
* 눈 감김 지속 시간
* 입 벌림 지속 시간
* 최근 30초 PERCLOS
* 최근 60초 하품 횟수
* 얼굴 방향 이탈 시간
* 최근 30초 평균 졸음 확률
* 현재 프레임의 CNN 예측 결과

---

## 17. 프로젝트 한계 및 개선 방향

본 프로젝트의 한계는 다음과 같다.

* Kaggle 데이터셋과 실제 웹캠 환경의 차이로 인한 domain gap 발생 가능
* 말하기나 웃는 동작이 하품으로 오인될 가능성
* 눈 크기, 안경, 카메라 각도에 따라 EAR 기준이 달라질 수 있음
* 모든 사용자에게 동일한 EAR/MAR 임계값이 적합하지 않을 수 있음
* 현재는 로컬 실행 중심이며, 공개 웹 서비스로 배포하려면 HTTPS 설정이 필요함

개선 방향은 다음과 같다.

* 사용자별 EAR/MAR 자동 보정 기능 추가
* 더 다양한 실제 웹캠 데이터 수집
* 졸음 상태를 normal / warning / drowsy 다중 클래스로 직접 학습
* LSTM 또는 Temporal CNN처럼 시간 정보를 반영하는 모델 적용
* 알림음, 팝업 경고, 진동 알림 기능 추가
* HTTPS 기반 웹 배포
