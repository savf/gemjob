# ![gemjob logo](gemjob_logo.png)

Gemjob is a tool to help clients optimize their job postings by giving them recommendations based on similar jobs.

## Run all microservices

```
docker-compose up
```

To run in background and detach from the log, use:

```
docker-compose up -d
```

You can then stop all containers with:

```
docker-compose stop
```

## Build individual microservices

```
docker-compose build webserver_module
docker-compose build data_mining_module
docker-compose build data_module
docker-compose build database_module
docker-compose build cluster_module
docker-compose build budget_module
docker-compose build feedback_module
docker-compose build jobtype_module
```

After building, the microservices can be individually started with:

```
docker-compose up webserver_module
docker-compose up data_mining_module
docker-compose up data_module
docker-compose up database_module
docker-compose up cluster_module
docker-compose up budget_module
docker-compose up feedback_module
docker-compose up jobtype_module
```
