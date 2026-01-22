import os


def get_settings():
    return {
        "database_url": os.getenv("DATABASE_URL", ""),
        "app_name": os.getenv("APP_NAME", "fastapi-factores"),
    }
