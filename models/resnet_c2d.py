# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import utils.logging as logging
from datasets.dataset_splits import DATASET_TO_NUM_CLASSES

logger = logging.get_logger(__name__)

class Classifier(nn.Module):
    """Classifier network.
    """
    def __init__(self, cfg):
        super(Classifier, self).__init__()

        if cfg.DATASETS[0] == "finegym":
            self.num_classes = cfg.EVAL.CLASS_NUM
        else:
            self.num_classes = DATASET_TO_NUM_CLASSES[cfg.DATASETS[0]]
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE

        self.fc_layers = []
        self.fc_layers.append(nn.Dropout(drop_rate))
        self.fc_layers.append(nn.Linear(self.embedding_size, self.num_classes))
        self.fc_layers = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        # Pass through fully connected layers.
        x = self.fc_layers(x)
        return x

class VanillaEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pooling = nn.AdaptiveMaxPool3d(1)
        fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE
        in_channels = cfg.MODEL.BASE_MODEL.OUT_CHANNEL
        self.fc_layers = []
        for channels, activate in fc_params:
            channels = channels*cap_scalar
            self.fc_layers.append(nn.Dropout(drop_rate))
            self.fc_layers.append(nn.Linear(in_channels, channels))
            self.fc_layers.append(nn.ReLU(True))
            in_channels = channels
        self.fc_layers = nn.Sequential(*self.fc_layers)
        self.embedding_layer = nn.Linear(in_channels, self.embedding_size)
    
    def forward(self, x, num_frames):
        batch_size, total_num_steps, c, h, w = x.shape
        num_context = total_num_steps // num_frames
        assert num_context == self.cfg.DATA.NUM_CONTEXTS
        x = x.view(batch_size * num_frames, num_context, c, h, w)
        x = x.transpose(1,2)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_frames, self.embedding_size)
        return x

class EmbedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        conv_params = cfg.MODEL.EMBEDDER_MODEL.CONV_LAYERS
        fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE
        in_channels = cfg.MODEL.BASE_MODEL.OUT_CHANNEL
        self.conv_layers = []
        for channels, kernel_size, tpad in conv_params:
            channels = channels*cap_scalar
            self.conv_layers.append(nn.Conv3d(in_channels, channels, kernel_size, padding=(tpad, 0, 0)))
            self.conv_layers.append(nn.BatchNorm3d(channels))
            self.conv_layers.append(nn.ReLU(True))
            in_channels = channels
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.pooling = nn.AdaptiveMaxPool3d(1)
        self.fc_layers = []
        for channels, activate in fc_params:
            channels = channels*cap_scalar
            self.fc_layers.append(nn.Dropout(drop_rate))
            self.fc_layers.append(nn.Linear(in_channels, channels))
            self.fc_layers.append(nn.ReLU(True))
            in_channels = channels
        self.fc_layers = nn.Sequential(*self.fc_layers)
        self.embedding_layer = nn.Linear(in_channels, self.embedding_size)

    def forward(self, x, num_frames):
        batch_size, total_num_steps, c, h, w = x.shape

        num_context = total_num_steps // num_frames
        assert num_context == self.cfg.DATA.NUM_CONTEXTS
        x = x.view(batch_size * num_frames, num_context, c, h, w)
        x = x.transpose(1,2)

        x = self.conv_layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_frames, self.embedding_size)
        return x

class MLPHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        projection_hidden_size = cfg.MODEL.PROJECTION_SIZE
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        self.net = nn.Sequential(nn.Linear(self.embedding_size, projection_hidden_size),
                                nn.BatchNorm1d(projection_hidden_size),
                                nn.ReLU(True),
                                nn.Linear(projection_hidden_size, self.embedding_size))
    
    def forward(self, x):
        b, l, c = x.shape
        x = x.view(-1,c)
        x = self.net(x)
        return x.view(b, l, c)

def load_simclr_pretrained(pretrained_weights):
    checkpoint = torch.load(pretrained_weights, map_location='cpu')
    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if 'num_batches_track' in key or 'momentum_encoder' in key: continue
        if 'encoder' in key:
            key = key.split('encoder.')[-1]
            key = key.replace('v1.weight', 'conv1.weight')
            key = key.replace('conconv1', 'conv1')
            key = key.replace('running_mean', 'running_mean')
            key = key.replace('running_var', 'running_var')
            state_dict[key] = value
    return state_dict

def load_byol_pretrained(pretrained_weights):
    checkpoint = torch.load(pretrained_weights, map_location='cpu')
    state_dict = {}
    for key, value in checkpoint['model'].items():
        if 'encoder_k' in key: continue
        if 'encoder' in key:
            key = key.split('module.encoder.')[-1]
            state_dict[key] = value
    return state_dict

def load_mocov2_pretrained(pretrained_weights):
    checkpoint = torch.load(pretrained_weights, map_location='cpu')
    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if 'encoder_q' in key:
            key = key.split('module.encoder_q.')[-1]
            state_dict[key] = value
    return state_dict

import os
def load_pretrained_resnet50(cfg, res50_model):
    if 'simclr' in cfg.MODEL.BASE_MODEL.NETWORK:
        # from https://github.com/PyTorchLightning/lightning-bolts
        pretrained_weights = os.path.join(cfg.args.workdir, "pretrained_models/simclr_imagenet.ckpt")
        state_dict = load_simclr_pretrained(pretrained_weights)
    elif 'byol' in cfg.MODEL.BASE_MODEL.NETWORK:
        pretrained_weights = os.path.join(cfg.args.workdir, "pretrained_models/BYOL_1000.pth")
        state_dict = load_byol_pretrained(pretrained_weights)
    elif 'mocov2' in cfg.MODEL.BASE_MODEL.NETWORK:
        pretrained_weights = os.path.join(cfg.args.workdir, "pretrained_models/moco_v2_200ep_pretrain.pth.tar")
        state_dict = load_mocov2_pretrained(pretrained_weights)
    else:
        pretrained_weights = os.path.join(cfg.args.workdir, "pretrained_models/resnet50-0676ba61.pth")
        state_dict = torch.load(pretrained_weights, map_location='cpu')

    msg = res50_model.load_state_dict(state_dict, strict=False)
    logger.info(msg)
    logger.info(f"=> loaded successfully '{pretrained_weights}'")


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        res50_model = models.resnet50(pretrained=False)
        if cfg.MODEL.BASE_MODEL.LAYER == 3:
            self.backbone = nn.Sequential(*list(res50_model.children())[:-3]) # output of layer3: 1024x14x14
            self.res_finetune = list(res50_model.children())[-3]
            cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 1024
        else:
            self.backbone = nn.Sequential(*list(res50_model.children())[:-2]) # output of layer4: 2048x7x7
            self.res_finetune = nn.Identity()
            cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
        if cfg.MODEL.EMBEDDER_TYPE == 'conv':
            self.embed = EmbedModel(cfg)
        elif cfg.MODEL.EMBEDDER_TYPE == 'vanilla':
            cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
            self.embed = VanillaEmbed(cfg)
        self.embedding_size = self.embed.embedding_size
        
        if cfg.MODEL.PROJECTION:
            self.ssl_projection = MLPHead(cfg)
        if cfg.TRAINING_ALGO == 'classification':
            self.classifier = Classifier(cfg)
        
    def forward(self, x, num_frames, video_masks=None, project=False, classification=False):

        batch_size, total_num_steps, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        if self.cfg.MODEL.TRAIN_BASE == 'frozen':
            self.backbone.eval()
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        if self.cfg.MODEL.EMBEDDER_TYPE == 'vanilla':
            x = self.res_finetune(x)
        _, c, h, w = x.shape
        x = x.view(batch_size, total_num_steps, c, h, w)

        x = self.embed(x, num_frames)

        if self.cfg.MODEL.PROJECTION and project:
            x = self.ssl_projection(x)
            x = F.normalize(x, dim=-1)
        elif self.cfg.MODEL.L2_NORMALIZE:
            x = F.normalize(x, dim=-1)
        if classification:
            return self.classifier(x)
        return x

