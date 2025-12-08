#!/bin/bash

docker-compose stop app

docker-compose build app

docker-compose up -d app