# Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning

**Pytorch** code for Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning, CVPR2022.

![pennaction_alignment](./pennaction_alignment.gif)

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

- Download the Pouring dataset at [pouring](https://drive.google.com/drive/folders/1hvA4bDqPnxjiVM4c4mxm-UPOuO1lAVxW?usp=sharing) 
- Download the PennAction dataset at [penn_action](https://drive.google.com/drive/folders/1hPbkKSSM5NoQKKzvAr2bHDp-pCHZpnVj?usp=sharing) 
- Download the FineGym dataset at [finegym](https://drive.google.com/drive/folders/1XOIy_6qtTo5MEecaWsi8X59p9BqLzmag?usp=sharing) 

BaiduCloud: https://pan.baidu.com/s/1Vu9Qkiei-O10tcdCJAwaHA  password: 7rbo

(Due to my limited storage, the link for finegym on google drive is expired. Only BaiduCloud link is avaliable now.)

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

#### Download ResNet50 pretrained with BYOL

Our ResNet50 beckbone is initialized with the weights trained by BYOL.

Download the pretrained weight at [pretrained_models](https://drive.google.com/drive/folders/1VwC4x5xj4Ho3bnh9wZZx--iYhIUguR-q?usp=sharing), and place it at `/home/username/datasets/pretrained_models`.



## Training

Check `./configs` directory to see all config settings.

##### Training on Pouring

Start training, assuming your machine only have one GPUs (if you have 4 GPUs, set `--nproc_per_node 4`):

```bash
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs/scl_transformer_config.yml --logdir ~/tmp/scl_transformer_logs
```

The config can be changed by adding `--opt TRAIN.BATCH_SIZE 1 TRAIN.MAX_EPOCHS 500`

Check the file `utils/config.py` to see all config options.

We use “automatic mixed precision training” by default, but it sometimes causes the `'nan' gradient` error. If you encounter this error, set `--opt USE_AMP false`.

##### Training on PennAction

```
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs/scl_transformer_action_config.yml --logdir ~/tmp/scl_transformer_action_logs
```



##### Training on FineGym

```
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs/scl_transformer_finegym_config.yml --logdir ~/tmp/scl_transformer_finegym_logs
```

Tips: The default number of data worker is 4, which might causes CPU overloaded for some Machines. In this case, you can set `--opt DATA.NUM_WORKERS 1`.

##### Pretraining on Kinetics400

Download K400 dataset from https://github.com/cvdfoundation/kinetics-dataset

```
python -m torch.distributed.launch --nproc_per_node 1 train.py --workdir ~/datasets --cfg_file ./configs/scl_transformer_k400_pretrain_config.yml --logdir ~/tmp/scl_transformer_k400_pretrain_logs
```



### Checkpoints

We provide the checkpoints trained by our CARL method at 

- [scl_transformer_logs](https://drive.google.com/drive/folders/163NvBvfb0_HyciMzqFuPCugN1F3oci7k?usp=sharing) (for Pouring)
- [scl_transformer_action_logs](https://drive.google.com/drive/folders/1RXD5Vl8hlsBPpiCaSIRlxEFUohPFCLUG?usp=sharing) (for PennAction)
- [scl_transformer_finegym_logs](https://drive.google.com/drive/folders/1XCxwo9KTXJBG3LfxQwikguLYOI6v-SHw?usp=sharing) (for FineGym). In this checkpoint, we also provide the extracted frame-wise representations of videos in FineGym.
-  [scl_transformer_k400_pretrain_logs](https://drive.google.com/drive/folders/1EYrpweUetE9I1oaia2qkhq2B7gITpR77?usp=sharing) (the model pretrained on K400 by our CARL)

Place these checkpoints at `/home/username/tmp` to evaluate them.

### Evaluation and Visualization

Start evaluation.

```bash
python -m torch.distributed.launch --nproc_per_node 1 evaluate.py --workdir ~/datasets --cfg_file ./configs/scl_transformer_config.yml --logdir ~/tmp/scl_transformer_logs
```

Tensorboard.

```bash
tensorboard --logdir=~/tmp/scl_transformer_logs
```

The video file of video alignment have already generated at `/home/username/tmp/scl_transformer_logs`



## Citation

```
@inproceedings{chen2022framewise,
      title={Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning}, 
      author={Minghao Chen and Fangyun Wei and Chong Li and Deng Cai},
      booktitle={CVPR},
      year={2022}
}
```



## Acknowledgment

The training setup code was modified from https://github.com/google-research/google-research/tree/master/tcc
