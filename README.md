# Puan DB

## Setup
### Basic
Puan DB requires one postgres database server. Start by setting the variable `POSTGRESQL_DATABASE_URL` in your environment file.

Poetry is used as dependency back end. Install it by
```bash
pip install poetry
```

Run the migrations by:
```bash
poetry run alembic upgrade head
```

Puan DB doesn't come with a solver out-of-the-box and requires a solver cluster or service to work properly. Most easy is to download and run the docker container `znittzel/puan-solver-orchestrator`. Make sure to set the `SOLVER_API_URL` variable in your environment file to point to the service.

Then run the project by:
```bash
poetry uvicorn main:app --host 0.0.0.0 --port 8000
```


### Using Docker
Or by using Docker compose:
```bash
docker compose up --build
```
This will download, build and run the necessary container services.

## Usage
Puan DB serves two purposes:
- Storing and providing combinatorial database models
- Different kinds of calculations on these models

Head over to `{service-url}:{service-port}/docs` to check docs.