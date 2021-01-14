# Continuum-Tensorflow
 
### A clean Tensorflow implementation of the Continual Learning library - [Continuum](https://github.com/Continvvm/continuum). 

*Effort has been made to retain the bare-bones TF/python calls without building too many abstractions on top of it. Abstractions make integrating native TF Dataset api's difficult without an investigation of the library's design. That has been avoided here.*


### Example:

Clone repo:
```
git clone https://github.com/aishikhar/continuum_tensorflow.git
```

Example:
```python
import tensorflow as tf
import numpy as np

train, test = continual_dataset(dataset = 'splitmnist', n_tasks = 5)

for task_no in range(n_tasks):

    task_label, data, labels = train[task_no]
    this_task = tf.data.Dataset.from_tensor_slices((data, labels)).batch(batch_size = 8)
    
    # Do your stuff
    learn_on(this_task)
```

### Tasks added:

CIFAR 100:

| Task 1             | Task 2  |      Task 3     |     Task 4    |     Task 5     |
:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
![](images/cifar10_0.jpg)  |  ![](images/cifar10_1.jpg) | ![](images/cifar10_2.jpg) | ![](images/cifar10_3.jpg) | ![](images/cifar10_4.jpg)



Split MNIST:

| Task 1             | Task 2  |      Task 3     |     Task 4    |     Task 5     |
:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
|<img src="images/mnist_0.jpg" width="150">|<img src="images/mnist_1.jpg" width="150">|<img src="images/mnist_1.jpg" width="150">|<img src="images/mnist_3.jpg" width="150">|<img src="images/mnist_4.jpg" width="150">|



Permuted MNIST:

| Task 1             | Task 2  |      Task 3     |     Task 4    |     Task 5     |
:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
|<img src="images/mnist_permuted_0.jpg" width="150">|<img src="images/mnist_permuted_1.jpg" width="150">|<img src="images/mnist_permuted_2.jpg" width="150">|<img src="images/mnist_permuted_3.jpg" width="150">|<img src="images/mnist_permuted_4.jpg" width="150">|

*(Source: [Continuum](https://github.com/Continvvm/continuum))*
