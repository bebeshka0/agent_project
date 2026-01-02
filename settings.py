import os


def get_postgres_connection_string() -> str:
    dsn: str | None = os.getenv("POSTGRES_CONNECTION_STRING")
    if dsn:
        return dsn

    user: str | None = os.getenv("POSTGRES_USER")
    password: str | None = os.getenv("POSTGRES_PASSWORD")
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: str = os.getenv("POSTGRES_PORT", "5432")
    database: str | None = os.getenv("POSTGRES_DB")

    if user and password and database:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    raise RuntimeError(
        "Postgres configuration is missing. Set POSTGRES_CONNECTION_STRING, or set "
        "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB (optionally POSTGRES_HOST, POSTGRES_PORT)."
    )


