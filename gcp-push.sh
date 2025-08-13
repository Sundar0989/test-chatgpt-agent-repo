#!/bin/bash

docker build . --platform linux/amd64 -t us-central1-docker.pkg.dev/atus-a4media-ds-dev/rapid-modeler/rapid_modeler:latest
docker push us-central1-docker.pkg.dev/atus-a4media-ds-dev/rapid-modeler/rapid_modeler:latest