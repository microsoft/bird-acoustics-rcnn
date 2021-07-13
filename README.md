# Recurrent Convolutional Neural networks for sound classification 

We present a deep learning approach towards the large-scale prediction and analysis of bird acoustics from 100 different birdspecies. We use spectrograms constructed on bird audio recordings from the Cornell Bird Challenge (CBC)2020 dataset, which includes recordings of multiple and potentially overlapping bird vocalizations per audio and recordings with background noise. Our experiments show that a hybrid modeling approach that involves a Convolutional Neural Network (CNN) for learning therepresentation for a slice of the spectrogram, and a Recurrent Neural Network (RNN) for the temporal component to combineacross time-points leads to the most accurate model on this dataset. The code has models ranging from stand-alone CNNs to hybrid models of various types obtained by combining CNNs with CNNs or RNNs of the following types:Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRU) and Legendre Memory Units (LMU).

## Setup

### Requirements
The code package is developed using Python 3.6 and Pytorch 1.2 with cuda 10.0. For running the experiments first install the required packages using 'requirements.txt'

## Experiments
The data for bird sound classification is downloaded from the Kaggle competition [Cornell birdcall Identification](https://www.kaggle.com/c/birdsong-recognition).

For running the experiments, a data preprocessing pipeline is demostrated in the process_data.ipynb

After preprocessing the data, the RCNN models with various combinations of representation/temporal models can be run as follows:

### CNN + CNN
An example is shown in CNN_CNN.ipynb notebook for the CNN and TCNN configs taken in the paper. In a similar way, a different set of configs could be supplied to the cnn+cnn model.
### CNN + RNN
An exampe for CNN+GRU, CNN+LMU, and CNN+LSTM is shown in CNN_RNN.ipynh notebook. Other variants of RCNNs with different set of parameters can be set as explained in the notebook.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
