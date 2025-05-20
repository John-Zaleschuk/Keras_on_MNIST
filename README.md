# Handwritten Digit Classification Using Keras

## Intro
This project trains a neural network to classify handwritten digits using the MNIST Dataset. Heres a great visualization of how the layers of a neural network break down and process the pixel data.

<p align="left">
  <img src="https://github.com/John-Zaleschuk/Keras_on_MNIST/blob/main/images/mnist_2layers.png" width="650"/>
</p>

*[Image Source](https://m-alcu.github.io/blog/2018/01/13/nmist-dataset/)*

The model uses Keras with TensorFlow to explore how structure and training parameters affect performance. The model is evaluated using 5-fold cross-validation and visualized with learning curves.

<br>

## High Epochs
The first implementation used a basic architecture with untuned hidden layers. While the model reached decent accuracy, it required a high number of training epochs to perform well. This showed promise but made it clear the configuration needed refinement.
<p align="left">
  <img src="https://github.com/John-Zaleschuk/Keras_on_MNIST/blob/main/images/almost_refined.png" width="650"/>
</p>
<br>

## Parameter Refinement
The final version fine-tuned the number of nodes and layers, allowing the network to converge faster while improving overall accuracy. It performs consistently well at reading handwritten digits and represents the completed version of this exercise.
<p align="left">
  <img src="https://github.com/John-Zaleschuk/Keras_on_MNIST/blob/main/images/refined.png" width="650"/>
</p>
