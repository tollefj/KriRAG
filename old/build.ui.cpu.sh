#!/bin/bash
docker build -t krirag-ui-cpu . -f krirag.cpu.dockerfile
docker save krirag-ui:latest > docker/krirag.tar