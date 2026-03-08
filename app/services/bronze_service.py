import io
from datetime import datetime

import pandas as pd

from app.core.config import settings
from app.services.s3_service import get_s3_client, object_exists


def build_bronze() -> dict:
    bucket = settings.s3_bucket
    raw_key = settings.raw_key
    bronze_key = settings.bronze_key

    if not object_exists(bucket, raw_key):
        raise FileNotFoundError(
            f"No existe el archivo raw en s3://{bucket}/{raw_key}"
        )

    s3 = get_s3_client()

    # Descargar CSV desde S3
    response = s3.get_object(Bucket=bucket, Key=raw_key)
    csv_bytes = response["Body"].read()

    # Leer CSV con pandas
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Convertir a parquet en memoria
    parquet_buffer = io.BytesIO()
    df.to_parquet(
        parquet_buffer,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    parquet_buffer.seek(0)

    # Subir parquet a S3
    s3.put_object(
        Bucket=bucket,
        Key=bronze_key,
        Body=parquet_buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    return {
        "status": "success",
        "bucket": bucket,
        "raw_key": raw_key,
        "bronze_key": bronze_key,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "executed_at": datetime.utcnow().isoformat(),
        "message": "Capa bronze creada correctamente.",
    }


def get_bronze_preview(limit: int = 10) -> dict:
    bucket = settings.s3_bucket
    bronze_key = settings.bronze_key

    if not object_exists(bucket, bronze_key):
        raise FileNotFoundError(
            f"No existe la capa bronze en s3://{bucket}/{bronze_key}"
        )

    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=bronze_key)
    parquet_bytes = response["Body"].read()

    df = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")
    preview_df = df.head(limit)

    return {
        "status": "success",
        "bucket": bucket,
        "bronze_key": bronze_key,
        "rows_total": int(df.shape[0]),
        "columns_total": int(df.shape[1]),
        "preview_rows": preview_df.to_dict(orient="records"),
    }