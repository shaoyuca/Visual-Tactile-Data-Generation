# Cross-Modal-Visual-Tactile-Data-Generation
This is the source code for our paper: 

Visual-Tactile Cross-Modal Data Generation using Residue-Fusion GAN with Feature-Matching and Perceptual Losses

![image](https://github.com/shaoyuca/Visual-Tactile-Data-Generation/blob/main/dataset/teaser.png)

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

Train the model (Users could set the number of epoch):
- Tactile-to-Visual (T2V)
```bash
pyhton CM_T2V.py --train --epoch <number>
```
- Visual-to-Tactile (V2T)
```bash
pyhton CM_V2T.py --train --epoch <number>
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

- Data: Training and testing data can be downloaded [here](https://drive.google.com/drive/folders/1J6G-KzMinu5XfAzQ2yzPUoy69ZcXWwEV?usp=sharing). After extracting the compressed file, put all the folders (from the downloaded folder) in the project directory './dataset' (the same directory where the teaser locates in).