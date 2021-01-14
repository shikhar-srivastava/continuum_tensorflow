import tensorflow as tf
import numpy as np
from continuum_tensorflow.learner.sample_learner import learn_on # sample learner
from continuum_tensorflow.data import continual_dataset # Primary CL data utils

n_tasks = 5

def example():
    
    # Create tasks
    tasks = continual_dataset(dataset = 'splitmnist', n_tasks = 5)
    train, test = tasks # train and test are lists of corresponding `n_tasks` diff. tasks

    # Iterate the `n_tasks`
    for task_no in range(n_tasks):

        # Retrieve tensors for each Task in the Dataset
        task_label, data, labels = train[task_no] # task: Task title, data: Tensor w. task data, label: Tensor w. labels

        # Create TF Dataset and run learner on task
        print('======= TASK : {} ======='.format(task_label))
        this_task = tf.data.Dataset.from_tensor_slices((data, labels)).take(1000).batch(batch_size = 8)
        
        # Do your stuff
        learn_on(this_task)

if __name__ == '__main__':

    example()
