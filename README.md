# Resnet_CIFAR10
The simple Resnet (Residual Neural Architecture) is a model consisting of 15 layers. Instead of directly connecting the residual layers, a convultaional block was utilized to connect the residual layers.

It was trained on the CIFAR10 dataset which is regarded as the basic testbench for CNNs. The model was trained for 200 epochs over 8 hours on a single GPU. A learning rate decay of 0.96 was used with ExponentialLR scheduler wrapping around the Adam optimizer. The model produeces train set accuracy about 90% and test set accuracy of 86.31%. Total number of parameters was 11.9 million.
