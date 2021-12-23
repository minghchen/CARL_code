# Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning

**Pytorch** code for Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning, CVPR2022.



## Requirements

```bash
# create conda env and install packages
conda create -y --name carl python=3.7.9

conda activate carl
# The code is tested on cuda10.1-cudnn7 and pytorch 1.6.0
conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install -y conda-build ipython pandas scipy pip av -c conda-forge

# install pip packages
pip install --upgrade pip
pip install -r requirements.txt
```



## Preparing Data

Create a directory to store datasets: 

```bash
mkdir /home/username/datasets
```

#### Download pre-processed datasets

- Download the Pouring dataset at [pouring](https://drive.google.com/drive/folders/1UIhHFVcUXL9CnKleB559_hnaphT524fF?usp=sharing) 
- Download the PennAction dataset at [penn_action](https://drive.google.com/drive/folders/1YO1BP8MCxtnWT8U2oMcdZOzN3KG7HwX8?usp=sharing) 
- Download the FineGym dataset at [finegym](https://drive.google.com/drive/folders/1XOIy_6qtTo5MEecaWsi8X59p9BqLzmag?usp=sharing) 

#### (Optionally) Pre-process datasets by yourself

Download Pouring using the script

```bash
sh dataset_preparation/download_pouring_data.sh
python dataset_preparation/tfrecords_to_videos.py
```

Download the original [ Penn Action](http://dreamdragon.github.io/PennAction/) dataset and [label files](https://drive.google.com/drive/folders/1rEnTfMopORljtEv6EGcNUKNEGTGj7_RZ). Run the preparation script:

```bash
python dataset_preparation/penn_action_to_tfrecords.py
python dataset_preparation/tfrecords_to_videos.py
```

Download the FineGym dataset from the official web [FineGym](https://sdolivia.github.io/FineGym/). Contact that author to get raw videos or using the youtube-dl script in `download_finegym_videos.py`.

Run the preparation script:

```bash
python dataset_preparation/finegym_process.py
```

We trim the raw video based on the event time-stamps in `finegym_annotation_info_v1.0.json`. Each event video is standardized to 640x360 resolution and 25 fps. We train the model on the event videos containing at least one sub-action. For further research, we also provide the event videos without sub-action labeled in `additional_processed_videos`.



## Training

Check `./configs` directory to see all config settings.

##### Training on Pouring

Start training, assuming your machine only have one GPUs (if you have 4 GPUs, set `--nproc_per_node 4`):

```bash
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs/simclr_transformer_config.yml --logdir ~/tmp/simclr_transformer_logs
```

The config can be changed by adding `--opt TRAIN.BATCH_SIZE 1 TRAIN.MAX_EPOCHS 500`

Check the file `utils/config.py` to see all config options.

We use “automatic mixed precision training” by default, but it sometimes causes the `'nan' gradient` error. If you encounter this error, set `--opt USE_AMP false`.

##### Training on PennAction

```
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs/simclr_transformer_action_config.yml --logdir ~/tmp/simclr_transformer_action_logs
```



##### Training on FineGym

```
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs/simclr_transformer_finegym_config.yml --logdir ~/tmp/simclr_transformer_finegym_logs
```

Tips: The default number of data worker is 4, which might causes CPU overloaded for some Machines. In this case, you can set `--opt DATA.NUM_WORKERS 1`.

##### Pretraining on Kinetics400

Download the dataset https://github.com/cvdfoundation/kinetics-dataset

```
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs/simclr_transformer_k400_pretrain_config.yml --logdir ~/tmp/simclr_transformer_k400_pretrain_logs
```



### Checkpoints

We provide the checkpoints trained by our CARL method at 

- [simclr_transformer_logs](https://drive.google.com/drive/folders/1N7Ez_SBgxP3rudYG9NGm5ulENWCE2E3G?usp=sharing) (for Pouring)
- [simclr_transformer_action_logs](https://drive.google.com/drive/folders/1Cfvd928dHZDW21ECDqF8zP5_V1VGoKPh?usp=sharing) (for PennAction)
- [simclr_transformer_finegym_logs](https://drive.google.com/drive/folders/1NhdWrL1lCMEzDKgDsHp7qbj2klIARrje?usp=sharing) (for FineGym). In this checkpoint, we also provide the extracted frame-wise representations of videos in FineGym.

Place these checkpoints at `/home/username/tmp` to evaluate them.

### Evaluation and Visualization

Start evaluation.

```bash
python -m torch.distributed.launch --nproc_per_node 1 evaluate.py --workdir ~/datasets --cfg_file ./configs/simclr_transformer_config.yml --logdir ~/tmp/simclr_transformer_logs
```

Tensorboard.

```bash
tensorboard --logdir=~/tmp/simclr_transformer_logs
```

The video file of video alignment have already generated at `/home/username/tmp/simclr_transformer_logs`



## Citation







## Acknowledgment

The training setup code was modified from https://github.com/google-research/google-research/tree/master/tcc
