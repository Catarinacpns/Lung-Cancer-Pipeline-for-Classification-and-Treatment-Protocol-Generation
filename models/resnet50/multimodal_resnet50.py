import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F
import os
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna

class MultiModalResNet(nn.Module):
    def __init__(
        self,
        num_classes_T,
        num_classes_N,
        num_classes_M,
        metadata_dim=4,
        dropout_rate=0.5,
        dropout_rate_RN50=0.3,
        trainable_layers=2,
        hidden_units=128,
        num_fc_layers=2
    ):
        super(MultiModalResNet, self).__init__()

        base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.cnn_out_dim = base_model.fc.in_features
        self.resnet_dropout = nn.Dropout(dropout_rate_RN50)

        for param in self.backbone.parameters():
            param.requires_grad = False

        resnet_blocks = ['layer1', 'layer2', 'layer3', 'layer4']
        selected_blocks = resnet_blocks[-trainable_layers:]
        for name, module in base_model.named_children():
            if name in selected_blocks:
                for param in module.parameters():
                    param.requires_grad = True

        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        combined_layers = []
        input_dim = self.cnn_out_dim + hidden_units
        for _ in range(num_fc_layers):
            combined_layers.append(nn.Linear(input_dim, hidden_units))
            combined_layers.append(nn.ReLU())
            combined_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_units
        self.combined_fc = nn.Sequential(*combined_layers)

        self.head_T = nn.Linear(hidden_units, num_classes_T)
        self.head_N = nn.Linear(hidden_units, num_classes_N)
        self.head_M = nn.Linear(hidden_units, num_classes_M)

    def forward(self, image, metadata):
        x_img = self.backbone(image).view(image.size(0), -1)
        x_img = self.resnet_dropout(x_img)
        x_meta = self.metadata_net(metadata)
        x = torch.cat((x_img, x_meta), dim=1)
        x = self.combined_fc(x)
        return {'T': self.head_T(x), 'N': self.head_N(x), 'M': self.head_M(x)}