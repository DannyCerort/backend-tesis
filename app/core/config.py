from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "anomalias-backend"
    aws_region: str = "us-east-2"
    s3_bucket: str
    raw_key: str
    bronze_key: str
    silver_key: str
    gold_series_key: str
    gold_forecast_key: str
    gold_ranking_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()