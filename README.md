# Soccer Analytics

This is a project that aims to get and analyze soccer performance, and potentially integrate machine learning to predict player performance. Right now require testing.

## The structure of the project (*planned*)

app/ \
 | data_service &emsp;&emsp;(*in working*) \
 |&emsp;&emsp;&emsp;| db/&emsp;&emsp;(store data in postgresql) \
 |&emsp;&emsp;&emsp;| cache/&emsp;&emsp;(store data in redis) \
 |&emsp;&emsp;&emsp;| data_service_factory.py&emsp;(the abstraction of get operation for db) \
 |&emsp;&emsp;&emsp;| data_type.py&emsp;&emsp;(defines all the available type of soccer data that db is able to get)
 |\
 |\
 |&emsp;&emsp;&emsp;| fetcher/&emsp;(fetch data from api)\
 |\
 | graphics/&emsp;&emsp;(generate graph from data: *in plan*)
 |\
 | ml_service/&emsp;&emsp;(*planned*)\
 | visualization_service/&emsp;(*planned*)\
 | frontend/&emsp;(planned)

## More details

Right now I am using [https://www.football-data.org/](https://www.football-data.org/) do get my data. Postgresql is my database and redis is my cache. \
The whole project is build in docker. \
The environment in docker is setted by .env file with the following attributes: \

```FOOTBALL_DATA_API_KEY
DATABASE_URL
REDIS_URL

#redis set up
REDIS_HOST
REDIS_PORT
REDIS_PASSWORD
REDIS_DB```\
