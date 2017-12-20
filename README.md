# keras-cnn-cifar10
#### Requirements
* keras
* numpy
* requests
* six
* tensorflow
* tqdm


#### Usage
```
git clone https://github.com/drewszurko/keras-cnn-cifar10.git
cd keras-cnn-cifar10/ 
python install -r requirements.txt
python setup.py install
python trainer/main.py
```
#### Dataset
The image data used in this example is from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. It will download automatically during installation. 

#### Dataset Description
* 50,000 32x32 color training images.
* 10,000 32x32 color testing images.
* 10 different classes.
* 6,000 images per class.
