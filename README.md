# Spark-Demo



Generic Command:
docker buildx build . --platform linux/amd64 --push -f Dockerfile.cuda -t $IMAGE_NAME

Example Command: 
docker buildx build . --platform linux/amd64 --push -f Dockerfile.cuda -t vivekkolasani1996/spark:3.3
