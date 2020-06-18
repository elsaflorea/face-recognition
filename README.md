# Face Recognition Streaming

Face recognition streaming to RTMP server with opencv

## Run application with docker

```bash
docker build --force-rm  --tag face_recognition_streaming:1.0.0 ./docker/
```

```bash
docker run -ti --rm \
    -v app:/home/root/workspace/app \
    --device=/dev/video0 \
    --name face_recognition_dev \
    rpi_face_recognition_dev:1.0.0 bash
```