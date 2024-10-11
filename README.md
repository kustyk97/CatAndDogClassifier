# CatAndDogClassifier
A simple implementation of a cat and dog classifier based on a neural network


# Docker 

Create Image

docker build -t custom_cuda_pytorch:1.0 .

Run docker 

docker run --gpus all --rm -it --name docker1 -p 8888:8888 -v ./:/workspace/ custom_cuda_pytorch:1.0

Run jupiter notebook inside dokcer

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser

