from pydantic_settings import BaseSettings

class EnvironmentVariables(BaseSettings):
    
    # Required settings
    POSTGRESQL_DATABASE_URL:    str
    SOLVER_API_URL:             str
    USERNAME:                   str
    PASSWORD:                   str

    # Optional settings
    VERSION:    str                     = "0.1.0"
    PORT:       int                     = 8000
    LOG_LEVEL:  str                     = "INFO"
    APP_NAME:   str                     = "Puan DB"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"