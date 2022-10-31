# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)

class DiffusionEmbedding(nn.Module):
  def __init__(self):
    super().__init__()
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, noise_level):
    x = self.build_embedding(noise_level)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x
    
  def build_embedding(self,noise_level):
    #steps: 0～63 每个数字/64 表示一条线上不同的噪声等级
    steps = torch.arange(64, dtype=noise_level.dtype, device=noise_level.device)
    #encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * steps.unsqueeze(0)+0.0001)
    #self.W = nn.Parameter(torch.randn(64)*30,requires_grad=False)
    encoding =  noise_level.unsqueeze(1) * 10.0**( steps * 4.0 / 63.0) 
    encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=1)
    return encoding



class ResidualBlock_init(nn.Module):
  def __init__(self, residual_channels, dilation):
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.output_projection = Conv1d(residual_channels, residual_channels, 1)
    self.output_residual = Conv1d(residual_channels, residual_channels, 1)

  def forward(self, x):
    y = self.dilated_conv(x)
    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)
    residual = self.output_residual(y)
    skip = self.output_projection(y)
    return (x + residual) / sqrt(2.0), skip


class InitialPredictor(nn.Module):
  def __init__(self,args,params):
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1)
    self.noisy_projection = Conv1d(1, params.residual_channels, 1)
    self.residual_layers = nn.ModuleList([
        ResidualBlock_init(params.residual_channels, 2**(i % params.dilation_cycle_length))
        for i in range(params.initial_predictor_reisdual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)
  
  def forward(self, noisy):
    x = noisy.unsqueeze(1)
    x = self.input_projection(x)
    x = F.relu(x)
    skip = []
    for layer in self.residual_layers:
      x, skip_connection = layer(x)
      skip.append(skip_connection)
    x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x.squeeze(1)

