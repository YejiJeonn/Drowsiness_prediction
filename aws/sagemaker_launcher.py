from __future__ import annotations

import argparse

import sagemaker
from sagemaker.pytorch import PyTorch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--prefix", default="drowsiness")
    parser.add_argument("--instance_type", default="ml.g4dn.xlarge")
    parser.add_argument("--model_name", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "mobilenet_v3_small"])
    args = parser.parse_args()

    session = sagemaker.Session()
    s3_data = f"s3://{args.bucket}/{args.prefix}/processed"
    s3_output = f"s3://{args.bucket}/{args.prefix}/sagemaker-outputs"

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",
        role=args.role,
        framework_version="2.1.0",
        py_version="py310",
        instance_count=1,
        instance_type=args.instance_type,
        output_path=s3_output,
        hyperparameters={
            "data_dir": "/opt/ml/input/data/processed",
            "model_name": args.model_name,
            "epochs": 15,
            "batch_size": 32,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "image_size": 224,
        },
    )

    estimator.fit({"processed": s3_data})
    print("Training job started/finished. Model artifacts:", estimator.model_data)


if __name__ == "__main__":
    main()
