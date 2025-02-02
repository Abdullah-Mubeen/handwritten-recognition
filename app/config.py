from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_name: str = "Handwritten Recognition"
    debug: bool = True

    class Config:
        env_file = "../.env"  # Loads environment variables from .env file

settings = Settings()
