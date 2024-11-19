#!/bin/bash
docker build -t krirag-ui . -f krirag.dockerfile
# docker save krirag-ui:latest > docker/krirag.tar