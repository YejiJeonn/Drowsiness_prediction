from __future__ import annotations

import argparse
from pathlib import Path

import boto3
from tqdm import tqdm


def upload_dir(local_dir: str, bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3")
    local_path = Path(local_dir)
    files = [p for p in local_path.rglob("*") if p.is_file()]
    for p in tqdm(files, desc=f"upload {local_dir} to s3://{bucket}/{prefix}"):
        key = f"{prefix.rstrip('/')}/{p.relative_to(local_path).as_posix()}"
        s3.upload_file(str(p), bucket, key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    upload_dir(args.local_dir, args.bucket, args.prefix)


if __name__ == "__main__":
    main()
