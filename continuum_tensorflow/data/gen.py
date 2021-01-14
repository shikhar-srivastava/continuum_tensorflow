import tensorflow as tf
import numpy as np


def continual_dataset(dataset: str, n_tasks: int) -> list:
    '''Sugar
        :param dataset: String specifying continual learning dataset
        :param n_tasks: Number of tasks to be created with dataset
    '''
    try:    
        dataset = dataset.lower()
        if('mnist' in dataset):
            if('split' in dataset):
                return get_mnist_split(n_tasks)
            elif(('permuted' in dataset) | ('perm' in dataset)):
                return get_mnist_permutations(n_tasks)
            else:
                raise ValueError('Badness :) in param `dataset` = {} .'.format(dataset))
        elif('cifar10' in dataset):   
            return get_cifar10_split(n_tasks)
        elif('cifar100' in dataset):   
            return get_cifar100_split(n_tasks)
        else:
            raise ValueError('Badness :) in param `dataset` = {} .'.format(dataset))
    
    except ValueError as e:
        print(e)



def get_mnist_permutations(n_tasks: int) -> list:
    ''' MNIST PERMUTATIONS Dataset
        :param n_tasks: Number of tasks to be created with dataset
    '''
    input_shape = [28,28,1]
    dtypes = tf.float32

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train, -1)
    x_train = tf.cast(x_train, dtype=dtypes) / 255
    x_test = tf.expand_dims(x_test, -1)
    x_test = tf.cast(x_test, dtype=dtypes) / 255

    task_permutation = []
    for _ in range(n_tasks):
        task_permutation.append(np.random.permutation(784))

    x_train = x_train.numpy()
    x_train = np.reshape(x_train, (x_train.shape[0],-1))
    x_test = x_test.numpy()
    x_test = np.reshape(x_test, (x_test.shape[0],-1))

    task_train = []
    task_test = []

    for task_no in range(n_tasks):
        data_train = np.reshape(x_train[:,task_permutation[task_no]], [x_train.shape[0]] + input_shape)
        data_test = np.reshape(x_test[:,task_permutation[task_no]], [x_test.shape[0]] + input_shape)
        task_train.append(['mnist_permutations',data_train, y_train])
        task_test.append(['mnist_permutations',data_test, y_test])

    return task_train, task_test

def get_mnist_raw():
    ''' MNIST
    '''
    input_shape = [28,28,1]
    dtypes = tf.float32

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train, -1)
    x_train = tf.cast(x_train, dtype=dtypes) / 255
    x_test = tf.expand_dims(x_test, -1)
    x_test = tf.cast(x_test, dtype=dtypes) / 255

    task_train = []
    task_test = []

    task_train.append(['raw', x_train, y_train])
    task_test.append(['raw', x_test, y_test])

    return task_train, task_test


def get_mnist_split(n_tasks: int) -> list:
    ''' SPLIT MNIST Dataset
        :param n_tasks: Number of tasks to be created with dataset
    '''
    input_shape = [28,28,1]
    dtypes = tf.float32

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train, -1)
    x_train = tf.cast(x_train, dtype=dtypes) / 255
    x_test = tf.expand_dims(x_test, -1)
    x_test = tf.cast(x_test, dtype=dtypes) / 255

    task_train = []
    task_test = []
    
    cpt = int(10 / n_tasks)

    for task_no in range(n_tasks):
        
        c1 = task_no * cpt
        c2 = (task_no + 1) * cpt

        i_tr = ((y_train >= c1) & (y_train < c2))
        i_te = ((y_test >= c1) & (y_test < c2))

        task_train.append([(c1, c2), x_train[i_tr], y_train[i_tr]])
        task_test.append([(c1, c2), x_test[i_te], y_test[i_te]])

    return task_train, task_test

def get_cifar10_split(n_tasks: int) -> list:
    ''' SPLIT CIFAR10 Dataset
        :param n_tasks: Number of tasks to be created with dataset
    '''
    input_shape = [32,32,3]
    dtypes = tf.float32

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = tf.cast(x_train, dtype=dtypes) / 255
    x_test = tf.cast(x_test, dtype=dtypes) / 255

    task_train = []
    task_test = []
    
    cpt = int(10 / n_tasks)

    for task_no in range(n_tasks):
        
        c1 = task_no * cpt
        c2 = (task_no + 1) * cpt

        i_tr = ((y_train >= c1) & (y_train < c2))
        i_te = ((y_test >= c1) & (y_test < c2))

        task_train.append([(c1, c2), x_train[i_tr[:,0]], y_train[i_tr[:,0]]])
        task_test.append([(c1, c2), x_test[i_te[:,0]], y_test[i_te[:,0]]])

    return task_train, task_test

def get_cifar100_split(n_tasks: int) -> list:
    ''' SPLIT CIFAR100 Dataset
        :param n_tasks: Number of tasks to be created with dataset
    '''
    input_shape = [32,32,3]
    dtypes = tf.float32

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = tf.cast(x_train, dtype=dtypes) / 255
    x_test = tf.cast(x_test, dtype=dtypes) / 255

    task_train = []
    task_test = []
    
    cpt = int(100 / n_tasks)

    for task_no in range(n_tasks):
        
        c1 = task_no * cpt
        c2 = (task_no + 1) * cpt

        i_tr = ((y_train >= c1) & (y_train < c2))
        i_te = ((y_test >= c1) & (y_test < c2))

        task_train.append([(c1, c2), x_train[i_tr[:,0]], y_train[i_tr[:,0]]])
        task_test.append([(c1, c2), x_test[i_te[:,0]], y_test[i_te[:,0]]])

    return task_train, task_test


def combine_tasks(task_data: list, n_tasks: int) -> tf.Tensor:
    '''
        Creates a combined tensor of all n_tasks within the task_data list. Returns type, data, label
            for combined task data
        :param task_data: list of data-tensors for n_tasks
        :param n_tasks: no of task data-tensors in task_data
        Returns:
            

    '''
    _type, _data, _label = task_data[0][0], task_data[0][1], task_data[0][2]
    for task_no in range(1, n_tasks):
        task_type, data, label = task_data[task_no][0], task_data[task_no][1], task_data[task_no][2]
        _data = tf.concat([_data, data], axis = 0)
        _type = tf.concat([_type, task_type], axis = 0)
        _label = tf.concat([_label, label], axis = 0)
    return _type, _data, _label