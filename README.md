# CNN Model Training with Only NumPy


This project is a simple training framework using only NumPy.
<br>
All training processes are implemeneted from scratch.
<br>
No deep learning frameworks such as PyTorch or TensorFlow are needed—just NumPy.


## Installation

```
pip install numpy==1.26
```


## Datasets: MNIST

Download MNIST datasets from [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)


## Training and Evaluation

### Run MNIST
```
python train.py --lr 1e-3 --epochs 10 --batch_size 256 --model_path /your/own/path (default=./model_ckpt) --data_path /your/own/path (default=./datasets)
```
### Test MNIST
```
python test.py --model_path /your/own/path (default=./model_ckpt) --data_path /your/own/path (default=./datasets)
```
