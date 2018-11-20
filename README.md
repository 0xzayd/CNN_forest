# Forest Damage Segmentation
## Description
Pixelwise classification for damaged areas in high resolution areal imagery of forests.
The input is a 4 bands orthophoto (RGB + NIR) of size 256x256 (can also be bigger and then divided into tiles of 256x256).
![Input Output Example](https://github.com/0xzayd/CNN_forest/blob/master/img/input_output.png)<br>
## U-Net Model
The segmentation model is based on the [U-Net](https://arxiv.org/pdf/1505.04597.pdf) architecture.<br>
The implemented model is divided into two parts: Encoding path and the Decoding path. <br>
The Encoding path decreases the dimensionality of the input image and learns to keep only the useful features<br>
The Decoding path increases the dimensionality of the activation maps from the encoding path and learns to recover the full spatial information regarding the label<br>.
![U-Net Architecture](https://github.com/0xzayd/CNN_forest/blob/master/img/model.png)<br>
<br>
Each Encoding block has the following layers:<br>
* Convolutional Layer + SeLU Activation
* Alpha Dropout to prevent Overfitting
* Convolutional Layer + SeLU Activation
* Maxpooling Layer (2,2)
<br>

Each Decoding block has the following layers:<br>
* Concatenation with the high level feature maps with the decoding block's input 
* Convolutional Layer + ReLU Activation
* Alpha Dropout to prevent Overfitting
* Convolutional Layer + SeLU Activation
* Transpose convolutional Layer to increase the dimensionality
<br>

## Data Preprocessing

10000x10000 pixels is divided into tiles of size 256x256<br>
Remote sensing data (from each band) is stored and stacked into an array of shape (256,256,4)<br>
Label data is converted to binary values (0 or 1) and stored in an array of shape (256,256,1)<br>

## Training
### Loss Function
The loss function used for the training i s the binary cross entropy <br>
L = -\frac{1}{N} \sum_{i = 1}^{N} [y_i log(\hat{y}_i) + (1 - y_i)log(1-\hat{y}_i)]
### Optimizer and Evaluation metrics
* Adam Optimizer
* Mean Intersection over Union (for monitoring the training using Tensorboard)