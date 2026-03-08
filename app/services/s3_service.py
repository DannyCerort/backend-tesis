import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


def get_s3_client():
    return boto3.client("s3", region_name=settings.aws_region)


def object_exists(bucket: str, key: str) -> bool:
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def get_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"