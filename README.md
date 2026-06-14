# AI Drowsiness Detection on AWS

Binary classification project: `drowsy` vs `normal`.

## Datasets

- Kaggle: `akashshingha850/mrl-eye-dataset`
- Kaggle: `enider/yawdd-dataset`
- Kaggle: `shuvokumarbasak4004/eye-open-close`

Label rule:

- `drowsy`: closed eye or yawn
- `normal`: open eye or no-yawn/normal

## 1. Colab setup

```bash
!pip install -r requirements.txt
```

Upload `kaggle.json` to Colab, then:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
bash scripts/download_kaggle.sh
python src/prepare_dataset.py --raw_dir data/raw --out_dir data/processed --image_size 224
python src/train.py --data_dir data/processed --output_dir outputs --model_name efficientnet_b0
```

Other models:

```bash
python src/train.py --data_dir data/processed --output_dir outputs_resnet18 --model_name resnet18
python src/train.py --data_dir data/processed --output_dir outputs_mobilenet --model_name mobilenet_v3_small
```

Prediction:

```bash
python src/predict.py --model_path outputs/best_model.pt --image_path path/to/test.jpg
```

## 2. AWS S3 upload

```bash
aws configure
aws s3 mb s3://YOUR_BUCKET_NAME
aws s3 sync data/processed s3://YOUR_BUCKET_NAME/drowsiness/processed
aws s3 sync outputs s3://YOUR_BUCKET_NAME/drowsiness/outputs
```

or using boto3:

```bash
python src/aws_upload.py --local_dir data/processed --bucket YOUR_BUCKET_NAME --prefix drowsiness/processed
python src/aws_upload.py --local_dir outputs --bucket YOUR_BUCKET_NAME --prefix drowsiness/outputs
```

## 3. SageMaker training

```bash
pip install sagemaker
python aws/sagemaker_launcher.py \
  --bucket YOUR_BUCKET_NAME \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE \
  --model_name efficientnet_b0
```

## 4. Local API test

```bash
python app.py
curl -X POST -F "file=@path/to/test.jpg" http://localhost:8080/predict
```

## 5. Docker API test

```bash
docker build -t drowsiness-api .
docker run -p 8080:8080 -e MODEL_PATH=/app/outputs/best_model.pt drowsiness-api
curl -X POST -F "file=@path/to/test.jpg" http://localhost:8080/predict
```
