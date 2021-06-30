# Cross-Modal-Visual-Tactile-Data-Generation
This is the source code for our paper: 

Cross-Modal Visual-Tactile Data Generation using Generative Adversarial Networks with Residue Fusion

![image](https://github.com/shaoyuca/Visual-Tactile-Data-Generation/blob/main/image-folder/teas.jpg)

## Setup

We run the program on a Linux desktop using python.

Environment requirements: 

- tensorflow 2.1.0  
- tensorlfow-addons 0.12.0  
- tensorlfow-io 0.17.0  
- librosa 0.8.0  
- scipy 1.4.1  
- opencv 4.5.1  

## Usage

- Train the model:
```bash
pyhton CM_T2V.py --train --epoch <number>
```

- Test the model:
```bash
python CM_T2V.py --test
```

- Visualize the generated frictional signals:
```bash
python CM_T2V.py --visualize
```

- Visualize the training processing:
```bash
cd logs
tensorboard --logdir=./
```

