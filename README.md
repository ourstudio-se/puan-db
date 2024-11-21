# Puan DB

## Setup
### Basic
Puan DB requires one postgres database server. Start by setting the variable `DATABASE_URL` in your environment file.

Poetry is used as dependency back end. Install it by
```bash
pip install poetry
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

### Simple optimization problem solving
Use `http://127.0.0.1:8000/api/v1/tools/search`

Puan DB serves main two purposes:
- Storing and providing combinatorial database models
- Different kinds of calculations on these models

Head over to `{service-url}:{service-port}/docs` to check docs.

### Static typing
Puan DB provides a static type based schema for a database. This provides rubustness to your database where each insert/update/deletion is validated with the database schema.

#### End points
- `GET /databases`: "Getting all databases"
- `POST /databases`: "Create a new database by given database name and schema in the body"
- `DELETE /databases/{name}`: "Delete database {name}"
- `PATCH /databases/{name}`: "Updates/replaces database schema with the one given"
- `GET /databases/{name}`: "Get the schema and data for database {name}"
- `GET /databases/{name}/schema`: "Get the schema for database {name}"
- `PATCH /databases/{name}/schema`: "Create or replace (bulk update) schema for database {name}"
- `GET /databases/{name}/schema/properties`: "Get the schema primitives for database {name}"
- `GET /databases/{name}/schema/primitives`: "Get the schema primitives for database {name}"
- `GET /databases/{name}/schema/composites`: "Get the schema primitives for database {name}"
- `GET /databases/{name}/schema/properties/{id}`: "Get the schema property {id} for database {name}"
- `GET /databases/{name}/schema/primitives/{id}`: "Get the schema primitive {id} for database {name}"
- `GET /databases/{name}/schema/composites/{id}`: "Get the schema composite {id} for database {name}"
- `PATCH /databases/{name}/schema/properties/{id}`: "Create or replace a schema property {id} for database {name}"
- `PATCH /databases/{name}/schema/primitives/{id}`: "Create or replace a schema primitive {id} for database {name}"
- `PATCH /databases/{name}/schema/composites/{id}`: "Create or replace a schema composite {id} for database {name}"
- `DELETE /databases/{name}/schema/properties/{id}`: "Delete a schema property {id} for database {name}"
- `DELETE /databases/{name}/schema/primitives/{id}`: "Create or replace a schema primitive {id} for database {name}"
- `DELETE /databases/{name}/schema/composites/{id}`: "Create or replace a schema composite {id} for database {name}"


- `GET /databases/{name}/data`: "Get the data for database {name}"
- `GET /databases/{name}/data/primitives`: "Get primitive's data for database {name}"
- `GET /databases/{name}/data/composites`: "Get composite's data for database {name}"
- `PATCH /databases/{name}/data`: "Create or replace (bulk update) data for database {name}"
- `PATCH /databases/{name}/data/primitives/{id}`: "Create or replace a primitive {id} for database {name}"
- `PATCH /databases/{name}/data/composites/{id}`: "Create or replace a composite {id} for database {name}"
- `DELETE /databases/{name}/data`: "Create or replace (bulk update) data for database {name}"
- `DELETE /databases/{name}/data/primitives/{id}`: "Create or replace a primitive {id} for database {name}"
- `DELETE /databases/{name}/data/composites/{id}`: "Create or replace a composite {id} for database {name}"



// Old
- `POST /databases/{name}/data/insert`: "Insert/append new data to database {name}. Data-schema validation is done before operation"
- `POST /databases/{name}/data/overwrite`: "Overwrite data with given. Data-schema validation is done before operation"
- `POST /databases/{name}/data/validate`: "Runs data-schema validation"
- `GET /databases/{name}/data/items/{id}`: "Get specific data element by ID"
- `PATCH /databases/{name}/data/items`: "Update specific data element (id is given in body element). Data-schema validation is done before saving"
- `DELETE /databases/{name}/data/items/{id}`: "Delete specific element. Data-schema validation is done before saving"

#### Feature improvements
- GET /databases/{name}/metadata: Consider adding an endpoint specifically for retrieving metadata about the database itself (like creation date, last updated, version, etc.), which can be useful for monitoring and insights.

- Bulk Operations:

        PATCH /databases/{name}/data/items/bulk: For bulk updates of specific elements, allowing IDs and data in the body, which can improve performance if multiple entries need to be updated.
        DELETE /databases/{name}/data/items/bulk: Similarly, bulk deletion by ID list could be useful, especially in large datasets.

- Data Search and Filter Endpoint:

        GET /databases/{name}/data/search: To allow filtering or searching of data elements based on attributes without pulling the full dataset, especially helpful in scenarios with large data sets.

- Versioning:

    If your database requires version tracking, endpoints like GET /databases/{name}/versions and POST /databases/{name}/versions/{version_id}/restore can be helpful for accessing or rolling back to previous versions.

- Logging and History:

    GET /databases/{name}/history: Retrieves a log of recent actions (like inserts, updates, deletes) for auditing purposes.
