
# Building a Simple Autoencoder - Lab

## Introduction
In this lab, we will try to build a simple autoencoder using Keras. We will work with the fashion-MNIST dataset to work out a problem of image compression and reconstruction. With a simple AE, the results may not be highly impressive, but the key takeaway from this lab is to see how the encoding/decoding functions are implemented neural nets and are differentiable with respect to the distance function. The differentiable part enables optimizing the parameters of the encoding/decoding functions to minimize the reconstruction loss.

Note: Refer to [Keras dcumentation](https://keras.io/) for details on methods used in this lab. 

## Objectives

You will be able to:

- Build a simple autoencoder in Keras
- Create the encoder and decoder functions as fully connected layers of a feed forward styled neural network. 
- Train an autoencoder with selected loss function and optimizer.

First let's import all the necessary libraries required for this experiment.


```python
# Install tensorflow and keras if you haven't done so already
# !pip install tensorflow
# !pip install keras

# Import necessary libraries



# Your code here


```

## The Fashion-MNIST dataset

We have already seen the popular MNIST dataset in our previous lessons. Let's load the very similar ["fashion-mnist" dataset](https://github.com/zalandoresearch/fashion-mnist). 

*"Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits."* 

This dataset comes packaged with keras and can be loaded using `fashion_mnist.load_data()`. More details on keras datasets can be seen on [keras documentation](https://keras.io/datasets/). Below is a quick sample of images that you may find in this dataset.

<img src="dataset.png" width=700>

Perform following tasks:
- Load the Fashion-mnist feature set into test and training datasets (ignore labels/targets for now)
- Normalize the values of train and test datasets between 0 and 1
- Check the shape of both datasets created above. 


```python



# Your code here


```




    ((60000, 28, 28), (10000, 28, 28))



Above we see that we have 3D arrays of train and test datasets containg 60K and 10K images of size 28x28 pixels. To work with the images as vectors, let’s reshape the 3D arrays as 2D matrices. 

- Reshape the 28 x 28 images into vectors of length 784 for both train and test set
- Print the shape of new datasets


```python


# Your code here


```




    ((60000, 784), (10000, 784))



## Build a Simple AutoEncoder

With our pre-processed data, we can start building a simple autoencoder with its The encoder and decoder functions are each __fully-connected__ neural layers. The encoder function will use a __ReLU__ (Rectified Linear Unit) activation function, while the decoder function uses a __sigmoid__ activation function.

[Here is a good reference on non-linear functions](https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f). 

> The encoder layer encodes the input image as a compressed, latent representation with reduced dimensionality. The decoder layer decodes the encoded image back to the original dimension. 

Here we will create the compressed representation with 32 dimensions with a __compression factor__  784 / 32 = 24.5

Let's build our Model . Perform following tasks.

- Define encoding dimensions (32) and calculate/print the compression factor
- Create a `Sequential()` autoencoder model in Keras

- Create a fully connected  __encoder layer__  to reduce the dimension from the original 784-dimensional vector to encoded 32-dimensional vector. Use the `relu` activation function

- Create a fully connected __decoder layer__ to restore the dimension from the encoded 32-dimensional representation back to the original 784-dimensional vector.Use `sigmoid` activation function

- Print he model summary 


```python


# Your code here


```

    Compression factor: 24.5
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_23 (Dense)             (None, 32)                25120     
    _________________________________________________________________
    dense_24 (Dense)             (None, 784)               25872     
    =================================================================
    Total params: 50,992
    Trainable params: 50,992
    Non-trainable params: 0
    _________________________________________________________________


## Inspect the Encoder
Let's try to examine how a compressed representation compares to the original image. We can extract the encoder model from the first layer of the autoencoder model created above. 

- Extract the first layer of autoencoder to create a new `encoder` model in Keras
- Show the summary of encoder model


```python


# Your code here


```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_5 (InputLayer)         (None, 784)               0         
    _________________________________________________________________
    dense_19 (Dense)             (None, 32)                25120     
    =================================================================
    Total params: 25,120
    Trainable params: 25,120
    Non-trainable params: 0
    _________________________________________________________________


This looks about right. We are now ready to train our autoencoder model. 

## Training the Model 

In order to train the model, We need to perform following tasks: 
- Compile the autoencoder model with `adam` optimization with `binary_crossentropy` loss (The purpose of the loss function is to reconstruct an image similar to the input image). 
- Fit the model with training dataset for both input and output (this implies image reconstruction)
- Iterate on the training data in batches of 256 in 20 epochs. Set `shuffle` to True for shuffling the batches.
- Use the test data for validation 

(Try increasing number of epochs and observe the effect on learning)


```python


# Your code here


```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 2s 36us/step - loss: 0.2835 - val_loss: 0.2856
    Epoch 2/20
    60000/60000 [==============================] - 2s 27us/step - loss: 0.2832 - val_loss: 0.2855
    Epoch 3/20
    60000/60000 [==============================] - 2s 27us/step - loss: 0.2831 - val_loss: 0.2853
    Epoch 4/20
    60000/60000 [==============================] - 2s 26us/step - loss: 0.2829 - val_loss: 0.2852
    Epoch 5/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2828 - val_loss: 0.2850
    Epoch 6/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2827 - val_loss: 0.2849
    Epoch 7/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2826 - val_loss: 0.2848
    Epoch 8/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2825 - val_loss: 0.2847
    Epoch 9/20
    60000/60000 [==============================] - 2s 30us/step - loss: 0.2824 - val_loss: 0.2846
    Epoch 10/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.2823 - val_loss: 0.2846
    Epoch 11/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2822 - val_loss: 0.2846
    Epoch 12/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2822 - val_loss: 0.2845
    Epoch 13/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2821 - val_loss: 0.2844
    Epoch 14/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2821 - val_loss: 0.2844
    Epoch 15/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2820 - val_loss: 0.2844
    Epoch 16/20
    60000/60000 [==============================] - 2s 29us/step - loss: 0.2819 - val_loss: 0.2843
    Epoch 17/20
    60000/60000 [==============================] - 2s 31us/step - loss: 0.2819 - val_loss: 0.2842
    Epoch 18/20
    60000/60000 [==============================] - 2s 31us/step - loss: 0.2819 - val_loss: 0.2842
    Epoch 19/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2818 - val_loss: 0.2842
    Epoch 20/20
    60000/60000 [==============================] - 2s 28us/step - loss: 0.2818 - val_loss: 0.2842





    <keras.callbacks.History at 0x182b015cc0>



Great, We’ve successfully trained our autoencoder. Our  autoencoder model can now compress a Fashion MNIST image down to 32 floating-point digits.

## Visualize The Results

To visually inspect the quality of compressed images, let's pick up a few images randomly and see how their  reconstruction looks. 

- Select 10 images randomly from the test set
- Uee the `encoder` model to predict encoded representation (the code) for chosen images
- Use the `autoencoder` model to get the reconstructed images
- For each image, show the actual image, the compressed representation and the reconstruction 


```python


# Your code here


```


![png](index_files/index_16_0.png)


We can see, as expected, the reconstructed images are quite lossy due to the huge reduction in dimensionality. We can see the shapes of these objects clearly, but the loss in image quality has taken away a lot of distinguishing features. So the compression is not highly impressive, but it works , and proves the point. We can improve the peroformance of such AEs using deeper networks as we shall see in our next lab. 


## Summary 

In this lab, we built a simple autoencoder using the fashion-MNIST dataset for a problem of image compression. We looked into creating the encoder and decoder layers in Keras and Training the model. We also visually inspected the results of this compression. 
