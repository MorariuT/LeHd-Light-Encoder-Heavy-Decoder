# LeHd: Light Encoder Heavy Decoder CNN  Architecture for Dynamic Video Streaming
## Abstract
Using Convolutional Neural Networks for video encoding is a continuously evolving field of research driven by the need of streaming high quality video from mobile devices in areas with variable network bandwidth, such as drones, space vehicles, and other off the grid equipment. This paper introduces LeHd, a fast and accurate approach to dynamically encode and decode video streams based on the real time available bandwidth using a UNet with a fine-tuned ResNet backbone for the encoder part. The novelty of this architecture consists in the light encoder that runs on the device and heavy decoder that is hosted on a GPU-capable server and the variable amount of residual matrices transmitted by the encoding device to the decoding-server. LeHd supports six operating modes, varying based on the available bandwidth, with dynamic switching between them. The architecture was evaluated on an aerial flight of Mars, simulating transition between transmission modes in real time. Experimental results showed that at highest compression (lowest bandwidth required) the image is still stable and defining features are still very distinguishable. On the encoding device, LeHd achieved an encoding rate of 30 frames per second which is suitable for most real time applications.

## Code 

### LeHd.py

Here we have the encoder and the decoder implemented. 

The encoder consists of a `renset18` pretrained model, that encodes each frame into a latent vector. While passing each frame through the `resnet` we save the residual paths for the decoder. 

The decoder is a Convolutional Neural Network that upscales the image from the latent vector size to the initial image size of $256x256$.

### PerceptualLoss.py

In this scirpt the loss fucntions are implemnted. To achieve the compression quality, we used a perceptual loss method. By computing the loss on the latents spaces after the images are passed through a `VGG16` network, we emphasise the structural quality of the image rather than the colors. By adding `MSE` and `FrequencyLoss` to the total loss, we can put accent on the colors aswell as the structure of the image.  

![arch](.screenshots/arch.png "Model Achitecture")
