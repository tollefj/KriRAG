#!/bin/bash
docker load -i docker/api.tar
docker load -i docker/ui.tar
docker-compose up -d