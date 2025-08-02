from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LangextractConfig(BaseSettings):
    API_KEY: SecretStr = Field(
        description="API key used to authenticate with the Langextract service"
    )
    MODEL_NAME: str = Field(
        default="gemini-2.5-flash", description="The name of the model"
    )

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LANGEXTRACT_",
        extra="ignore",
    )


config = LangextractConfig()
