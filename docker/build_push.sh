cp ../dev_requirements.txt . && \
docker build --no-cache -t supervisely/projections-service:1.0.2 . && \
rm dev_requirements.txt && \
docker push supervisely/projections-service:1.0.2
