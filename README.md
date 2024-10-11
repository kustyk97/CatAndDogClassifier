# CatAndDogClassifier
A simple implementation of a cat and dog classifier based on a neural network


# Docker 

Create Image
```bash
docker build -t custom_cuda_pytorch:1.0 .
```
Run docker 
```bask
docker run --gpus all --rm -it --name docker1 -p 8888:8888 -v ./:/workspace/ custom_cuda_pytorch:1.0
```
Run jupiter notebook inside dokcer
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```
# Train model
You can train model by train.py script.
```bash
python train.py --trainSet path/to/trainset --testSet path/to/testset
```

# Test model
To test model you can run test.py script.
```bash
python test.py --model path/to/model --dataset path/to/dataset
```

# Predict 
You can use model to predict class of an image by predict.py script.
```bash
python predict.py --model path/to/model --image path/to/image
```

