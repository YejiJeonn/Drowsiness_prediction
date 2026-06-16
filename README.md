> 하품 횟수, 눈 감김 시간, PERCLOS, 얼굴 방향, CNN 모델의 졸음 확률을 활용하여 웹캠 기반 졸음 상태를 실시간으로 판단하는 컴퓨터비전 프로젝트입니다.
> 

## GitHub Repository

``` https://github.com/YejiJeonn/Drowsiness_prediction ``` 

---

## 1. 문제 정의

본 프로젝트는 운전자 졸음 방지, 생산 공정 휴먼에러 방지, 학생의 맞춤형 학습계획 수립 등에 도움이 될 수 있도록 웹캠 영상을 기반으로 사용자의 졸음 및 집중도 저하 상태를 실시간으로 감지하는 것을 목표로 한다.

단일 이미지 한 장만으로 졸음 여부를 판단하는 방식은 실제 서비스에서 한계가 있다. 예를 들어 눈을 한 번 깜빡인 순간의 사진만 보고 졸음이라고 판단하면 오탐이 발생할 수 있고, 반대로 하품이나 장시간 눈 감김과 같은 시간적 패턴은 단일 프레임만으로 충분히 판단하기 어렵다.

따라서 본 프로젝트에서는 웹캠 프레임을 일정 간격으로 분석하고, 최근 30~60초 동안의 프레임 기록을 누적하여 최종 상태를 판단한다.

최종 상태는 다음 세 단계로 구분하였다.

| 상태 | 의미 |
| --- | --- |
| `normal` | 정상 상태 |
| `warning` | 졸음 의심 상태 |
| `drowsy` | 졸음 감지 상태 |

주요 기능은 다음과 같다.

- 웹캠 기반 실시간 졸음 감지
- CNN 모델을 이용한 `normal / drowsy` 이진 분류
- MediaPipe Face Mesh 기반 눈 감김, 입 벌림, 얼굴 방향 분석
- 최근 30초 기준 PERCLOS 계산
- 눈 감김 지속 시간 측정
- 최근 60초 기준 하품 횟수 측정
- 정상, 졸음 의심, 졸음 감지 상태 표시
- 모델별 성능 비교
- Flask 기반 웹캠 분석 API 구현

---

## 2. 전체 시스템 구조

```
[사용자 웹캠]
     ↓
[Browser: JavaScript frame capture]
     ↓
[Flask API: /analyze_frame]
     ↓
┌─────────────────────────────────────────┐
│ CNN Image Classification Model          │
│ - MobileNetV3 / ResNet18 / EfficientNet │
│ - normal / drowsy probability           │
└─────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────┐
│ MediaPipe Face Mesh                  │
│ - Eye Aspect Ratio (EAR)             │
│ - Mouth Aspect Ratio (MAR)           │
│ - Face direction                     │
└──────────────────────────────────────┘
     ↓
┌──────────────────────────────────────┐
│ Temporal Analysis                    │
│ - Eye closed seconds                 │
│ - PERCLOS over 30 seconds            │
│ - Yawn count over 60 seconds         │
│ - Face away seconds                  │
│ - Average drowsy probability         │
└──────────────────────────────────────┘
     ↓
[Final Status Decision]
     ↓
normal / warning / drowsy
```

---

## 3. 사용 기술

| 구분 | 기술 | 사용 이유 |
| --- | --- | --- |
| Language | Python | 전체 데이터 처리, 학습, 서버 구현 |
| Deep Learning | PyTorch | CNN 모델 학습 및 추론 |
| Computer Vision | Torchvision | 이미지 전처리, 데이터 로딩, pretrained vision model 사용 |
| Computer Vision | OpenCV | 영상 데이터에서 프레임 추출 |
| Image Processing | PIL/Pillow | 이미지 열기, RGB 변환, resize, 저장 |
| Face Landmark | MediaPipe Face Mesh | 눈, 입, 얼굴 방향 분석을 위한 얼굴 랜드마크 추출 |
| Web Server | Flask | 웹캠 프레임을 서버로 받아 분석하는 API 구현 |
| Frontend | HTML/CSS/JavaScript | 웹캠 실행 및 분석 결과 표시 |
| Cloud | AWS EC2 | 모델 학습 및 서버 실행 환경 |

### Torchvision

Torchvision은 PyTorch에서 컴퓨터비전 작업을 쉽게 수행하기 위한 라이브러리이다.
본 프로젝트에서는 MobileNetV3, ResNet18, EfficientNet-B0 모델을 불러오고, 이미지 resize, normalization, augmentation, ImageFolder 데이터 로딩에 사용하였다.

### OpenCV

OpenCV는 이미지와 영상을 처리하기 위한 대표적인 컴퓨터비전 라이브러리이다.
본 프로젝트에서는 YawDD와 같은 영상 데이터에서 일정 간격으로 프레임을 추출하기 위해 사용하였다.
영상 파일을 열고, 특정 프레임을 읽고, 이미지로 변환하는 과정이 컴퓨터비전 전처리에 해당하므로 OpenCV를 Computer Vision 기술로 분류하였다.

### PIL/Pillow

PIL은 Python Imaging Library의 약자이며, 현재는 Pillow 라이브러리로 사용된다.
본 프로젝트에서는 이미지 파일을 열고 RGB 형식으로 변환한 뒤, 224×224 크기로 조정하고 JPG 파일로 저장하는 데 사용하였다.

---

## 4. AWS 활용 내용

본 프로젝트에서는 AWS EC2를 모델 학습 및 서버 실행 환경으로 활용하였다.

| AWS 요소 | 활용 내용 |
| --- | --- |
| EC2 | 데이터 다운로드, 전처리, 모델 학습, Flask 서버 실행 |
| EBS Storage | Kaggle 원본 데이터, 전처리 데이터, 학습 결과 저장 |
| Security Group | SSH 접속 및 API 포트 접근 제어 |
| Public IP | EC2 서버 외부 접속 |
| SSH | 로컬 환경에서 EC2 접속 |
| Port Forwarding | EC2 Flask 서버를 로컬 브라우저에서 테스트 |
| S3 | 학습 결과 및 데이터 업로드 확장 가능 |

AWS를 사용한 이유는 다음과 같다.

- Kaggle 데이터셋 용량이 커서 서버 환경에서 안정적으로 관리하기 위함
- 전처리와 모델 학습을 로컬 PC에 의존하지 않고 EC2에서 수행하기 위함
- 학습 결과인 `best_model.pt`, `metrics.json`, `confusion_matrix.png` 등을 서버에 저장하기 위함
- Flask API 서버를 실행하여 실제 서비스 구조를 테스트하기 위함
- 추후 S3 업로드, HTTPS 배포, 도메인 연결 등으로 확장하기 위함

웹캠 API는 일반 IP 주소보다 `localhost` 또는 HTTPS 환경에서 안정적으로 동작하므로, 최종 시연은 로컬 포트포워딩 방식으로 진행하였다.

---

## 5. 사용 데이터셋

본 프로젝트에서는 Kaggle에서 제공되는 졸음, 눈 감김, 하품 관련 데이터셋을 사용하였다.

- MRL 데이터셋 (Awake 42,952 / Sleepy 41,946)
https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset
- Eye Open Close Dataset (눈 감은 이미지 10,000 / 눈 뜬 이미지 10,000 / 256x256 픽셀 사이즈로 조정한 눈 감은 이미지 10,000)
https://www.kaggle.com/datasets/shuvokumarbasak4004/eye-open-close
- YawDD Dataset (348개의 영상 데이터)
https://www.kaggle.com/datasets/enider/yawdd-dataset

| 데이터셋 | 데이터 형태 | 사용 목적 |
| --- | --- | --- |
| MRL Eye Dataset | 눈 이미지 | 눈 뜸/눈 감김 상태 학습 |
| Eye Open Close Dataset | 눈 이미지 | 눈 상태 데이터 보강 |
| YawDD Dataset | 하품/비하품 영상 | 하품 상태 데이터 확보 및 프레임 추출 |

최종 라벨은 다음과 같이 이진 분류로 통합하였다.

| 원본 상태 | 최종 라벨 |
| --- | --- |
| open eye, no_yawn, normal, awake | `normal` |
| closed eye, yawn, sleepy, drowsy | `drowsy` |

즉, 눈을 뜨고 있거나 하품하지 않는 상태는 `normal`, 눈을 감고 있거나 하품하는 상태는 `drowsy`로 분류하였다.

---

## 6. 데이터 병합 및 라벨링 방식

여러 데이터셋은 `data/raw` 폴더 아래에 저장한 뒤, 전처리 코드가 전체 파일을 재귀적으로 탐색하였다.

처리 대상 파일 확장자는 다음과 같다.

- 이미지: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- 영상: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

라벨은 파일명과 폴더명에 포함된 키워드를 기준으로 자동 추론하였다.

normal 라벨 키워드:

```
no_yawn, no-yawn, noyawn, no yawn, not_yawn, normal, awake, open, open_eye, open-eyes, openeyes, opened
```

drowsy 라벨 키워드:

```
closed, close, closed_eye, closed-eyes, closedeyes, sleep, sleepy, drowsy, yawn, yawning
```

전처리 코드에서는 normal 키워드를 먼저 검사하고, 그 다음 drowsy 키워드를 검사하였다.

그 이유는 `no_yawn`이라는 단어 안에 `yawn`이 포함되어 있기 때문이다. 만약 drowsy 키워드를 먼저 검사하면 `no_yawn` 데이터가 잘못해서 drowsy로 분류될 수 있다.

라벨 키워드가 없는 파일은 normal/drowsy로 확실하게 분류할 수 없기 때문에 제외하였다.

---

## 7. 영상 데이터 처리 방식

YawDD와 같은 영상 데이터는 영상 전체를 직접 학습하지 않고, 일정 간격으로 프레임을 추출하여 이미지 데이터로 변환하였다.

전처리 실행 설정은 다음과 같다.

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

```
1. OpenCV의 cv2.VideoCapture로 영상 파일을 연다.
2. 전체 프레임 수를 확인한다.
3. 0, 300, 600, 900 ... 번째 프레임을 선택한다.
4. 영상 하나당 최대 20장까지만 추출한다.
5. OpenCV의 BGR 이미지를 RGB 이미지로 변환한다.
6. 224×224 크기로 resize한다.
7. 임시 폴더에 JPG 이미지로 저장한다.
8. 이후 이미지 데이터와 동일하게 train/val/test 데이터셋에 병합한다.
```

영상의 모든 프레임을 사용하지 않은 이유는 다음과 같다.

- 영상의 모든 프레임을 사용하면 데이터 수가 과도하게 증가함
- 연속 프레임은 서로 매우 비슷해 중복 데이터가 많음
- 특정 영상이 전체 데이터셋에서 과도한 비중을 차지할 수 있음
- 학습 시간과 저장 공간이 크게 증가함

따라서 본 프로젝트에서는 영상 하나당 최대 20장까지만 추출하여 학습 효율을 높였다.

---

## 8. 전처리에서 제외한 데이터

전처리 과정에서 다음 데이터는 제외하였다.

| 제외 대상 | 제외 이유 |
| --- | --- |
| 라벨 키워드가 없는 파일 | normal/drowsy로 라벨링할 수 없음 |
| 이미지/영상 확장자가 아닌 파일 | 학습에 사용하지 않는 파일 형식 |
| 깨진 이미지 또는 열 수 없는 이미지 | PIL에서 로딩 실패 |
| 영상의 모든 프레임 | 중복이 많고 데이터가 과도하게 증가함 |
| 영상 하나당 20장을 초과하는 프레임 | 특정 영상이 데이터셋을 과도하게 차지하지 않도록 제한 |
| 클래스당 3,000장을 초과하는 데이터 | 클래스 균형 유지 및 학습 시간 단축 |

---

## 9. 최종 데이터 구성

클래스 불균형을 줄이기 위해 `normal`과 `drowsy`의 개수를 동일하게 맞춘 균형 데이터셋을 구성하였다.

본 프로젝트에서는 클래스당 최대 3,000장을 사용하였다.

| Label | Count |
| --- | --- |
| drowsy | 3,000 |
| normal | 3,000 |
| total | 6,000 |

데이터는 train/validation/test로 70:15:15 비율로 나누었다. 또한 `stratify` 옵션을 사용하여 각 split에서 normal과 drowsy 비율이 동일하게 유지되도록 하였다.

| Split | drowsy | normal | total |
| --- | --- | --- | --- |
| train | 2,100 | 2,100 | 4,200 |
| validation | 450 | 450 | 900 |
| test | 450 | 450 | 900 |
| total | 3,000 | 3,000 | 6,000 |

최종 데이터셋 구조는 다음과 같다.

```
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

현재 전처리 방식은 `normal / drowsy` 클래스 균형은 보장하지만, 원본 데이터셋별 동일 비율 샘플링까지 보장하지는 않는다. 따라서 향후 개선 방향으로는 원본 데이터셋별 stratified sampling을 적용할 수 있다.

---

## 10. 이미지 전처리 및 증강

이미지에는 다음 공통 전처리를 적용하였다.

```
1. 이미지 열기
2. RGB 형식으로 변환
3. 224×224 크기로 resize
4. Tensor 변환
5. ImageNet mean/std 기준 normalization
```

ImageNet mean/std 기준 normalization은 ImageNet으로 사전학습된 모델이 학습 당시 사용한 입력 분포와 유사하게 맞추기 위한 과정이다. 본 프로젝트에서 사용한 모델은 ImageNet pretrained weight를 사용하므로, 입력 이미지도 ImageNet 기준 평균과 표준편차로 정규화하였다.

학습 데이터에는 일반화 성능을 높이기 위해 데이터 증강을 적용하였다.

| 증강 기법 | 설명 | 적용 이유 |
| --- | --- | --- |
| RandomHorizontalFlip | 이미지를 무작위로 좌우 반전 | 얼굴 방향이 좌우로 달라지는 상황 대응 |
| RandomRotation | 이미지를 일정 각도 범위 내에서 회전 | 고개 기울어짐이나 카메라 각도 변화 대응 |
| ColorJitter | 밝기, 대비, 색감 등을 무작위로 변화 | 조명 변화와 카메라 색감 차이 대응 |

Validation과 Test 데이터에는 증강을 적용하지 않고 resize와 normalization만 적용하였다. 검증 및 테스트 데이터는 실제 성능 평가용이므로 원본 분포를 유지해야 하기 때문이다.

---

## 11. 사용 모델 및 선택 기준

본 프로젝트에서는 3개의 CNN 모델을 비교하였다.

| 모델 | 선택 이유 |
| --- | --- |
| MobileNetV3-Small | 가볍고 빠른 모델로 실시간 웹캠 추론에 적합 |
| ResNet18 | CNN 이미지 분류에서 널리 사용되는 안정적인 baseline 모델 |
| EfficientNet-B0 | 정확도와 연산량의 균형이 좋은 모델 |

모델은 실시간 웹캠 기반 졸음 감지 서비스라는 목적에 맞춰 선정하였다. 따라서 단순히 정확도가 높은 모델 하나만 선택한 것이 아니라, 정확도, 추론 속도, 모델 크기, 비교 실험 가능성을 기준으로 선정하였다.

---

## 12. Transfer Learning 및 모델 구조 변경

모든 모델은 ImageNet pretrained weight를 기반으로 사용하였다.

ImageNet pretrained weight란 ImageNet이라는 대규모 이미지 데이터셋으로 미리 학습된 모델 가중치를 의미한다. 이를 사용하면 모델이 선, 모서리, 질감, 색상 변화 등 기본적인 시각 특징을 이미 학습한 상태에서 시작할 수 있다.

본 프로젝트는 모델을 처음부터 학습한 것이 아니라, ImageNet pretrained model을 기반으로 Transfer Learning을 적용하였다.

기존 pretrained 모델은 ImageNet의 1,000개 클래스를 분류하도록 마지막 classifier layer가 구성되어 있다.
하지만 본 프로젝트는 `normal`과 `drowsy` 두 클래스를 분류하는 이진 분류 문제이므로, 마지막 classifier layer를 출력 노드 2개짜리 Linear layer로 교체하였다.

모델별 수정 방식은 다음과 같다.

```python
# ResNet18
model.fc = nn.Linear(in_features, 2)

# EfficientNet-B0
model.classifier[1] = nn.Linear(in_features, 2)

# MobileNetV3-Small
model.classifier[-1] = nn.Linear(in_features, 2)
```

즉, 앞쪽 CNN feature extractor는 ImageNet에서 학습된 특징 추출 능력을 활용하고, 마지막 분류 계층만 본 프로젝트의 `normal / drowsy` 분류에 맞게 수정하여 fine-tuning하였다.

---

## 13. 학습 설정 및 파라미터

공통 학습 설정은 다음과 같다.

| 항목 | 내용 |
| --- | --- |
| 입력 크기 | 224×224 RGB |
| 출력 클래스 | drowsy, normal |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Best Model 기준 | Validation Accuracy |
| 평가 데이터 | Test Set |
| Pretrained | True |
| Feature Extract | False |

`feature_extract=false`는 마지막 classifier만 학습한 것이 아니라, pretrained model 전체를 fine-tuning했다는 의미이다.

모델별 학습 파라미터는 다음과 같다.

| Model | Epochs | Batch Size | Learning Rate | Weight Decay | Image Size | Pretrained | Feature Extract |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | 15 | 64 | 0.0005 | 0.00005 | 224 | True | False |
| ResNet18 | 12 | 32 | 0.0001 | 0.0001 | 224 | True | False |
| EfficientNet-B0 | 15 | 32 | 0.0003 | 0.0001 | 224 | True | False |

### CrossEntropyLoss 사용 이유

본 프로젝트는 `normal`과 `drowsy` 중 하나를 예측하는 이미지 분류 문제이다. 모델은 이미지 한 장에 대해 두 개의 출력값, 즉 `drowsy` 점수와 `normal` 점수를 출력한다.

CrossEntropyLoss는 모델의 클래스별 출력값과 실제 정답 라벨을 비교하여 분류 오차를 계산하는 대표적인 손실 함수이므로 본 프로젝트에 적합하다.

### AdamW 사용 이유

AdamW는 Adam 계열 optimizer의 적응적 학습률 장점을 유지하면서 weight decay를 분리해 적용하는 optimizer이다. 본 프로젝트에서는 pretrained CNN 모델을 fine-tuning하므로 안정적인 학습과 과적합 완화를 위해 AdamW를 사용하였다.

---

## 14. 성능 평가 지표

모델 성능은 Test Set을 기준으로 평가하였다.

평가 지표는 다음과 같다.

| 지표 | 의미 |
| --- | --- |
| Accuracy | 전체 샘플 중 정확히 예측한 비율 |
| Precision Macro | 각 클래스 precision의 평균 |
| Recall Macro | 각 클래스 recall의 평균 |
| F1 Macro | precision과 recall의 조화 평균 |
| Confusion Matrix | 실제 라벨과 예측 라벨의 관계 |
| FPS | 초당 처리 가능한 이미지 수 |
| Inference Time | 이미지 1장 처리에 걸리는 평균 시간 |

실시간 웹캠 서비스에서는 정확도뿐만 아니라 FPS와 inference time도 중요하다. 따라서 본 프로젝트에서는 모델별 분류 성능과 추론 속도를 함께 비교하였다.

---

## 15. 모델별 성능 결과

모델별 성능 결과는 다음과 같다.

| Model | Accuracy | Precision | Recall | F1-score | FPS | Inference Time |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | 0.9733 | 0.9737 | 0.9733 | 0.9733 | 3361.35 | 0.0003s |
| ResNet18 | 0.9744 | 0.9747 | 0.9744 | 0.9744 | 1310.76 | 0.0008s |
| EfficientNet-B0 | 0.9678 | 0.9678 | 0.9678 | 0.9678 | 883.69 | 0.0011s |

성능 결과를 보면 ResNet18이 가장 높은 Accuracy와 F1-score를 기록하였다. MobileNetV3-Small은 ResNet18보다 정확도는 약간 낮지만 FPS가 가장 높아 실시간 추론에 가장 유리하다. EfficientNet-B0는 일반적으로 정확도와 연산량의 균형이 좋은 모델이지만, 본 실험에서는 상대적으로 낮은 성능과 가장 낮은 FPS를 보였다.

따라서 실제 웹캠 서비스 관점에서는 MobileNetV3-Small이 속도 측면에서 가장 유리하고, 분류 정확도만 보면 ResNet18이 가장 좋은 결과를 보였다.

---

## 16. 성능 결과 해석

세 모델 모두 96% 이상의 높은 성능을 보였다. 이는 다음과 같은 이유로 해석할 수 있다.

- `normal / drowsy` 이진 분류 문제로 단순화되어 클래스 차이가 비교적 명확함
- 눈 뜸, 눈 감김, 하품은 시각적 특징 차이가 큼
- ImageNet pretrained 모델을 사용하여 적은 데이터로도 좋은 특징 추출 가능
- 클래스 균형을 맞춰 학습하여 한쪽 클래스 편향을 줄임
- 학습 데이터에 증강을 적용하여 조명, 회전, 좌우 방향 변화에 대한 일반화 성능을 높임

반대로 실제 웹캠 환경에서는 Kaggle 데이터셋과 다른 조명, 각도, 해상도, 얼굴 크기, 안경 착용 여부 등에 따라 성능이 달라질 수 있다.

따라서 최종 서비스에서는 CNN의 단일 프레임 예측만 사용하지 않고, EAR, MAR, PERCLOS, 하품 횟수, 얼굴 방향 이탈 시간과 같은 시간 기반 지표를 함께 사용하였다.

---

## 17. PERCLOS 설명

PERCLOS는 `Percentage of Eye Closure` 계열의 졸음 감지 지표로, 일정 시간 동안 눈이 감겨 있었던 비율을 의미한다.

본 프로젝트에서는 실제 동공이 가려진 비율을 직접 측정하지 않고, MediaPipe Face Mesh로 계산한 EAR 값이 임계값보다 낮은 프레임을 눈 감김 프레임으로 간주하였다. 그리고 최근 30초 동안의 눈 감김 프레임 비율을 PERCLOS 기반 지표로 사용하였다.

계산 방식은 다음과 같다.

```
PERCLOS = 최근 30초 동안 눈 감김 프레임 수 / 최근 30초 전체 프레임 수
```

예를 들어 최근 30초 동안 60개의 프레임을 분석했고, 그중 21개 프레임에서 눈이 감겨 있었다면 다음과 같다.

```
PERCLOS = 21 / 60 = 0.35 = 35%
```

PERCLOS는 단일 프레임이 아니라 일정 시간 동안의 눈 감김 비율을 보기 때문에 졸음 판단에 적합하다.

---

## 18. 실시간 상태 판단 기준

본 프로젝트의 실시간 서비스는 단일 프레임만으로 졸음 여부를 결정하지 않는다.

각 프레임에서 다음 정보를 계산한다.

- CNN 모델의 `drowsy` 확률
- 눈 감김 여부
- 입 벌림 여부
- 얼굴 방향 이탈 여부

이후 최근 30~60초 동안의 결과를 누적하여 최종 상태를 결정한다.

### 18.1 눈 감김 기준

MediaPipe Face Mesh 랜드마크를 이용해 눈의 세로/가로 비율인 EAR을 계산한다.

```
EAR = 눈 세로 거리 / 눈 가로 거리
```

현재 기준은 다음과 같다.

```
EAR < 0.21 → 눈 감김
```

| 조건 | 상태 |
| --- | --- |
| 눈 감김 2초 이상 지속 | 졸음 감지 |
| 최근 30초 PERCLOS 25% 이상 | 졸음 의심 |
| 최근 30초 PERCLOS 35% 이상 | 졸음 감지 |

### 18.2 하품 기준

입의 세로/가로 비율인 MAR을 계산한다.

```
MAR = 입 세로 거리 / 입 가로 거리
```

현재 기준은 다음과 같다.

```
MAR > 0.32 → 입 벌림
입 벌림 상태가 1초 이상 지속 → 하품 1회
```

입이 계속 벌어진 상태에서는 중복 카운트하지 않는다. 입이 닫힌 뒤 다시 1초 이상 벌어졌을 때 새로운 하품으로 카운트한다.

| 조건 | 상태 |
| --- | --- |
| 최근 60초 하품 2회 이상 | 졸음 의심 |
| 최근 60초 하품 3회 이상 | 졸음 감지 |

### 18.3 얼굴 방향 기준

얼굴 중심과 코 위치의 차이를 이용해 정면을 보고 있는지 판단하였다.

| 조건 | 상태 |
| --- | --- |
| 정면 이탈 5초 이상 지속 | 졸음 감지 또는 집중도 저하 |

### 18.4 CNN 평균 졸음 확률 기준

CNN 모델은 각 프레임에 대해 `normal`과 `drowsy` 확률을 출력한다. 이 값은 단일 프레임만으로 최종 판단하지 않고, 최근 30초 평균값으로 사용하였다.

| 조건 | 상태 |
| --- | --- |
| 최근 30초 평균 졸음 확률 60% 이상 | 졸음 의심 |
| 최근 30초 평균 졸음 확률 70% 이상 + PERCLOS 25% 이상 | 졸음 감지 |

---

## 19. 최종 상태 결정 기준

최종 상태는 세 단계로 구분하였다.

| 상태 | 의미 | 기준 |
| --- | --- | --- |
| `normal` | 정상 | 위험 조건 없음 |
| `warning` | 졸음 의심 | 평균 졸음 확률 60% 이상, PERCLOS 25% 이상, 최근 60초 하품 2회 이상 중 하나 |
| `drowsy` | 졸음 감지 | 눈 감김 2초 이상, PERCLOS 35% 이상, 최근 60초 하품 3회 이상, 얼굴 이탈 5초 이상, 평균 졸음 확률 70% 이상 + PERCLOS 25% 이상 중 하나 |

즉, CNN 모델의 단일 프레임 예측은 보조 지표로 사용하고, 최종 알림은 눈 감김 지속 시간, PERCLOS, 하품 횟수, 얼굴 방향 이탈 시간과 같은 시간 기반 지표를 함께 고려하여 결정하였다.

---

## 20. 실행 방법

### 20.1 데이터 전처리

```bash
python src/prepare_dataset_fast.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --image_size 224 \
  --video_every_n_frames 300 \
  --max_video_frames_per_video 20 \
  --max_per_class 3000
```

### 20.2 모델 학습

```bash
python src/train.py \
  --data_dir data/processed \
  --output_dir outputs_mobilenet \
  --model_name mobilenet_v3_small \
  --num_workers 2
```

```bash
python src/train.py \
  --data_dir data/processed \
  --output_dir outputs_resnet18 \
  --model_name resnet18 \
  --num_workers 2
```

```bash
python src/train.py \
  --data_dir data/processed \
  --output_dir outputs_efficientnet \
  --model_name efficientnet_b0 \
  --num_workers 2
```

### 20.3 웹캠 서비스 실행

```bash
export MODEL_PATH=outputs_efficientnet/best_model.pt
python app.py
```

브라우저에서 접속한다.

```
<http://localhost:8080>
```

웹캠 시작 버튼을 누르면 프레임 분석이 시작되고, 화면에 다음 값들이 표시된다.

- 현재 상태
- 눈 감김 지속 시간
- 입 벌림 지속 시간
- 최근 30초 PERCLOS
- 최근 60초 하품 횟수
- 얼굴 방향 이탈 시간
- 최근 30초 평균 졸음 확률
- 현재 프레임의 CNN 예측 결과

---

## 21. 학습 결과 파일 및 로그

학습 결과는 각 모델별 output 폴더에 저장된다.

```
outputs_mobilenet/
├── best_model.pt
├── metrics.json
├── training_log.csv
└── confusion_matrix.png

outputs_resnet18/
├── best_model.pt
├── metrics.json
├── training_log.csv
└── confusion_matrix.png

outputs_efficientnet/
├── best_model.pt
├── metrics.json
├── training_log.csv
└── confusion_matrix.png
```

학습 로그는 각 모델별로 다음 파일에 저장하였다.

- `train_mobilenet.log`
- `train_resnet18.log`
- `train_efficientnet.log`
- `train_test.log`

각 로그에는 epoch별 train loss, validation loss, validation accuracy 등이 기록되며, 이를 통해 학습 과정과 best model 저장 시점을 확인할 수 있다.

### 학습 로그 및 metrics 일부 예시

### MobileNetV3-Small

```json
{
  "fps": 3361.3520114837174,
  "model_name": "mobilenet_v3_small",
  "class_to_idx": {
    "drowsy": 0,
    "normal": 1
  },
  "idx_to_class": {
    "0": "drowsy",
    "1": "normal"
  },
  "hyperparameters": {
    "epochs": 15,
    "batch_size": 64,
    "lr": 0.0005,
    "weight_decay": 0.00005,
    "image_size": 224,
    "pretrained": true,
    "feature_extract": false
  }
}
```

### ResNet18

```json
{
  "fps": 1310.757345990985,
  "model_name": "resnet18",
  "class_to_idx": {
    "drowsy": 0,
    "normal": 1
  },
  "idx_to_class": {
    "0": "drowsy",
    "1": "normal"
  },
  "hyperparameters": {
    "epochs": 12,
    "batch_size": 32,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "image_size": 224,
    "pretrained": true,
    "feature_extract": false
  }
}
```

### EfficientNet-B0

```json
{
  "fps": 883.6916536501661,
  "model_name": "efficientnet_b0",
  "class_to_idx": {
    "drowsy": 0,
    "normal": 1
  },
  "idx_to_class": {
    "0": "drowsy",
    "1": "normal"
  },
  "hyperparameters": {
    "epochs": 15,
    "batch_size": 32,
    "lr": 0.0003,
    "weight_decay": 0.0001,
    "image_size": 224,
    "pretrained": true,
    "feature_extract": false
  }
}
```

---

## 22. Code 구조

```
Drowsiness_prediction/
├── app.py
├── templates/
│   └── index.html
├── src/
│   ├── prepare_dataset_fast.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── modeling.py
│   └── aws_upload.py
├── scripts/
│   └── download_kaggle.sh
├── aws/
│   └── sagemaker_launcher.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs_mobilenet/
│   ├── best_model.pt
│   ├── metrics.json
│   ├── training_log.csv
│   └── confusion_matrix.png
├── outputs_resnet18/
│   ├── best_model.pt
│   ├── metrics.json
│   ├── training_log.csv
│   └── confusion_matrix.png
├── outputs_efficientnet/
│   ├── best_model.pt
│   ├── metrics.json
│   ├── training_log.csv
│   └── confusion_matrix.png
├── train_mobilenet.log
├── train_resnet18.log
├── train_efficientnet.log
├── train_test.log
├── requirements.txt
├── Dockerfile
└── README.md
```

각 주요 파일의 역할은 다음과 같다.

| 파일 | 역할 |
| --- | --- |
| `src/prepare_dataset_fast.py` | 원본 데이터 탐색, 라벨 추론, 영상 프레임 추출, 데이터 분할 |
| `src/modeling.py` | 모델 생성 및 이미지 transform 정의 |
| `src/train.py` | 모델 학습, 검증, 테스트 평가, 결과 저장 |
| `src/predict.py` | 단일 이미지 예측 |
| `app.py` | Flask API 및 실시간 웹캠 졸음 분석 |
| `templates/index.html` | 웹캠 화면 및 분석 결과 표시 |
| `scripts/download_kaggle.sh` | Kaggle 데이터셋 다운로드 |
| `outputs_*/metrics.json` | 모델별 성능 지표 |
| `outputs_*/confusion_matrix.png` | 모델별 confusion matrix |
| `train_*.log` | 모델별 학습 로그 |

---

## 23. 프로젝트 한계 및 개선 방향

본 프로젝트의 한계는 다음과 같다.

- Kaggle 데이터셋과 실제 웹캠 환경의 차이로 인한 domain gap 발생 가능
- 말하기나 웃는 동작이 하품으로 오인될 가능성
- 눈 크기, 안경, 카메라 각도에 따라 EAR 기준이 달라질 수 있음
- 모든 사용자에게 동일한 EAR/MAR 임계값이 적합하지 않을 수 있음
- 현재 데이터 병합 방식은 클래스 균형은 맞추지만, 원본 데이터셋별 균형을 완전히 보장하지는 않음
- 현재는 로컬 실행 중심이며, 공개 웹 서비스로 배포하려면 HTTPS 설정이 필요함

개선 방향은 다음과 같다.

- 사용자별 EAR/MAR 자동 보정 기능 추가
- 더 다양한 실제 웹캠 데이터 수집
- 졸음 상태를 `normal / warning / drowsy` 다중 클래스로 직접 학습
- LSTM 또는 Temporal CNN처럼 시간 정보를 반영하는 모델 적용
- 알림음, 팝업 경고, 진동 알림 기능 추가
- HTTPS 기반 웹 배포
- 원본 데이터셋별 stratified sampling 적용

---

## 24. 핵심 요약

본 프로젝트는 단순 이미지 분류를 넘어서, 실제 웹캠 환경에서 사용할 수 있도록 시간 기반 졸음 판단 로직을 추가하였다.

CNN 모델은 현재 프레임의 졸음 확률을 예측하고, MediaPipe Face Mesh는 눈, 입, 얼굴 방향 정보를 추출한다. 이후 최근 프레임 기록을 누적하여 PERCLOS, 눈 감김 지속 시간, 하품 횟수 등을 계산함으로써 더 현실적인 졸음 감지 시스템을 구현하였다.

모델 비교 결과 ResNet18이 가장 높은 분류 성능을 보였고, MobileNetV3-Small은 가장 높은 FPS를 보여 실시간 서비스에 유리한 모델임을 확인하였다.
