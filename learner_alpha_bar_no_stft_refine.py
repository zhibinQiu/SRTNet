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

import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import from_path as dataset_from_path
from model_alpha_bar_no_stft import DiffuSE
from model_initial_predictor import InitialPredictor


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffuSELearner:
  def __init__(self, model_dir, model,initial_predictor, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.initial_predictor = initial_predictor
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True
    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)**0.5
    noise_level = np.concatenate([[1.0], noise_level], axis=0)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.loss_fn2 = nn.L1Loss()
    self.summary_writer = None
  
  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
      intial_predictor_state = self.initial_predictor.module.state_dict()
    else:
      model_state = self.model.state_dict()
      intial_predictor_state = self.initial_predictor.state_dict()
    
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'intial_predictor':{ k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in intial_predictor_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }


  def load_state_dict(self, state_dict):

    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
        self.model.module.load_state_dict(state_dict['model'])
        self.initial_predictor.module.load_state_dict(state_dict['intial_predictor'])
    else:
        self.model.load_state_dict(state_dict['model'])
        self.initial_predictor.load_state_dict(state_dict['intial_predictor'])
          
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']
    

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
  
    torch.save(self.state_dict(), save_name)
    
    if os.path.islink(link_name):
        os.unlink(link_name)
    os.symlink(save_basename, link_name)
      
  def restore_from_checkpoint(self, filename='weights'):
     
      try:
        #pdb.set_trace()
        checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
        self.load_state_dict(checkpoint)
        return True
      except FileNotFoundError:
        return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if  torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
              self._write_summary(self.step, features, loss)
          if self.step % len(self.dataset) == 0:
              self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None
    for param in self.initial_predictor.parameters():
      param.grad = None
      
    audio = features['audio']
    noisy = features['noisy']    
    N, T = audio.shape
    device = audio.device
    S = len(self.params.noise_schedule)
    self.noise_level = self.noise_level.to(device)
   
    with self.autocast:
      s = torch.randint(0, S+1, [N], device=audio.device)
      l_a,l_b = self.noise_level[s-1],self.noise_level[s]
      noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
      m = (((1-noise_scale**2)/noise_scale)**0.5)
      noise = torch.randn_like(audio)    
      noise_scale = noise_scale.unsqueeze(1)
      m = m.unsqueeze(1)
      # 做残差操作
      y_init = self.initial_predictor(noisy)
      ############################refine###########################
      noisy = noisy - y_init
      audio = audio -y_init
      noisy_audio = (1-m) * noise_scale  * audio + m * noise_scale * noisy  + (1.0 - (1+m**2) * noise_scale ** 2)**0.5 * noise
      combine_noise = (m * noise_scale * (noisy-audio) + (1.0 - (1+m**2) *noise_scale**2)**0.5 * noise) / ((1-noise_scale**2)**0.5+1e-8)
      predicted = self.model(noisy_audio, noisy, noise_scale.squeeze(1))
      ############################refine###########################
      
      ############################direct###########################
      # noisy_audio = (1-m) * noise_scale  * audio + m * noise_scale * y_init  + (1.0 - (1+m**2) * noise_scale ** 2)**0.5 * noise
      # # 实际的噪声
      # combine_noise = (m * noise_scale * (y_init-audio) + (1.0 - (1+m**2) *noise_scale**2)**0.5 * noise) / ((1-noise_scale**2)**0.5+1e-8)
      # predicted = self.model(noisy_audio, y_init, noise_scale.squeeze(1))
      ############################direct###########################
      
    loss = self.loss_fn(combine_noise, predicted.squeeze(1))
    
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    nn.utils.clip_grad_norm_(self.initial_predictor.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('train/loss', loss, step)
    #writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer
        
def _train_impl(replica_id, model,initial_predictor, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam([ {'params': model.parameters()},
        {'params': initial_predictor.parameters()}], lr=params.learning_rate,eps=1e-4)
  learner = DiffuSELearner(args.model_dir, model,initial_predictor, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint(args.pretrain_path)
  learner.train(max_steps=args.max_steps)


def train(args, params):
  dataset = dataset_from_path(args.clean_dir, args.noisy_dir, args.data_dirs, params, se=args.se, voicebank=args.voicebank)
  model = DiffuSE(args, params).cuda()
  initial_predictor = InitialPredictor(args,params).cuda()
  _train_impl(0, model,initial_predictor,dataset, args, params)

def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  #torch.autograd.set_detect_anomaly(True)
  model = DiffuSE(args, params).to(device)
  initial_predictor = InitialPredictor(args,params).cuda()
  model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=True)
  initial_predictor = DistributedDataParallel(initial_predictor, device_ids=[replica_id], find_unused_parameters=True)
  _train_impl(replica_id, model,initial_predictor, dataset_from_path(args.clean_dir, args.noisy_dir, args.data_dirs, params, se=args.se, voicebank=args.voicebank, is_distributed=True), args, params)
