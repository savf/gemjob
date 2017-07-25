# <img src="gemjob_logo.jpg" width="516px"/>

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
docker-compose build knn_module
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
docker-compose up knn_module
docker-compose up budget_module
docker-compose up feedback_module
docker-compose up jobtype_module
```

## Run for development or production

For development purposes (code mounted instead of added into container and more ports accessible from the docker host) a docker-compose.override.yml has been defined, which will be executed when `docker-compose` is run without any arguments.

For production, an additional docker-compose.production.yml has been defined, which needs to be explicitly called with `docker-compose`:

```
docker-compose -f docker-compose.yml -f docker-compose.production.yml ...
```