

# A2C-Collab

This is the official code repository for the paper "Integrating Advantage Actor-Critic in Multi-Robot Collaboration".

## Overview

This repository contains the implementation of the A2C-Collab framework presented in our paper.

## Setup



### setup conda env and package install



```
conda create -n a2c_collab python=3.8 
conda activate a2c_collab
```



### Install mujoco and dm_control



```
pip install mujoco==2.3.0
pip install dm_control==1.0.8 
```

### Install other packages



```
pip install -r requirements.txt
```

### Usage

```
python run_dialog.py --task cabinet -llm Llama-3.1-70B-Instruct --skip_display
python run_dialog.py --task sort -llm Llama-3.1-70B-Instruct --skip_display
python run_dialog.py --task sweep -llm Llama-3.1-70B-Instruct --skip_display
python run_dialog.py --task rope -llm Llama-3.1-70B-Instruct --skip_display
python run_dialog.py --task sandwich -llm Llama-3.1-70B-Instruct --skip_display
```

###Model change
Please check the model_manager.py to change the default model as we have listed in the main paper

