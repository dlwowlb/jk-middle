import boto3
from pathlib import Path

def upload_to_s3(bucket: str, prefix: str, local_dir: str):
    """Upload all .tar files in local_dir to s3://{bucket}/{prefix}/."""
    s3 = boto3.client("s3")
    for tar_path in Path(local_dir).glob("*.tar"):
        s3.upload_file(str(tar_path), bucket, f"{prefix}/{tar_path.name}")

# 사용 예시


if __name__ == "__main__":
    upload_to_s3(
        bucket="jk1b1ucket",
        prefix="datasets/webdataset/audio",
        local_dir="./webdataset"
    )
