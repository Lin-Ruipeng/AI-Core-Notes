# Optimizing Model Parameters

## 超参数

Hyperparameters are adjustable parameters that let you control the model optimization process. Different hyperparameter values can impact model training and convergence rates

We define the following hyperparameters for training:

 - **Number of Epochs** - the number of times to iterate over the dataset

 - **Batch Size** - the number of data samples propagated through the network before the parameters are updated

 - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

## 概念

Each iteration of the optimization loop is called an **epoch**.

Each epoch consists of two main parts:

 - **The Train Loop** - iterate over the training dataset and try to converge to optimal parameters.

 - **The Validation/Test Loop** - iterate over the test dataset to check if model performance is improving.

## 优化器

Inside the training loop, optimization happens in three steps:

 - Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.

 - Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.

 - Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.


