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
import librosa
import random
from argparse import ArgumentParser
from collections import defaultdict
from params import AttrDict, params as base_params
from model_alpha_bar_no_stft import DiffuSE
from model_initial_predictor import InitialPredictor
from pesq import pesq
import speechmetrics
from glob import glob
from tqdm import tqdm
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
from utils import parse_csv
from matlib_eval import eval_composite
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
random.seed(213)
models = {}

def load_model(model_dir=None, args=None, params=None, device=torch.device('cuda')):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model = DiffuSE(args, AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model
  model = models[model_dir]
  model.params.override(params)
  return model
def load_initial_predictor(model_dir,args=None, params=None, device=torch.device('cuda')):
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    initial_predictor = InitialPredictor(args, AttrDict(base_params)).to(device)
    initial_predictor.load_state_dict(checkpoint['intial_predictor'])
    initial_predictor.eval()
    return initial_predictor
def inference_schedule(model, fast_sampling=False):
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) 
    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)
    beta = inference_noise_schedule
    print('beta:{}'.format(beta))
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    print('alpha_cum:{}'.format(alpha_cum))
    sigmas = [0 for i in alpha]
    for n in range(len(alpha) - 1, -1, -1): 
      sigmas[n] = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])
      
    m = [0 for i in alpha] 
    gamma = [0 for i in alpha] 
    delta = [0 for i in alpha]  
    d_x = [0 for i in alpha]  
    d_y = [0 for i in alpha]  
    delta_cond = [0 for i in alpha]  
    delta_bar = [0 for i in alpha] 
    c1 = [0 for i in alpha] 
    c2 = [0 for i in alpha] 
    c3 = [0 for i in alpha] 
    oc1 = [0 for i in alpha] 
    oc3 = [0 for i in alpha] 
    
    for n in range(len(alpha)):
      m[n] = min(((1- alpha_cum[n])/(alpha_cum[n]**0.5)),1)**0.5
    m[-1] = 1    

    for n in range(len(alpha)):
      delta[n] = max(1-(1+m[n]**2)*alpha_cum[n],0)
      gamma[n] = sigmas[n]

    for n in range(len(alpha)):
      if n >0: 
        d_x[n] = (1-m[n])/(1-m[n-1]) * (alpha[n]**0.5)
        d_y[n] = (m[n]-(1-m[n])/(1-m[n-1])*m[n-1])*(alpha_cum[n]**0.5)
        delta_cond[n] = delta[n] - (((1-m[n])/(1-m[n-1])))**2 * alpha[n] * delta[n-1]
        delta_bar[n] = (delta_cond[n])* delta[n-1]/ delta[n]
      else:
        d_x[n] = (1-m[n])* (alpha[n]**0.5)
        d_y[n]= (m[n])*(alpha_cum[n]**0.5)
        delta_cond[n] = 0
        delta_bar[n] = 0 
    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)
    
    for n in range(len(alpha)):
      oc1[n] = 1 / alpha[n]**0.5
      oc3[n] = oc1[n] * beta[n] / (1 - alpha_cum[n])**0.5
      if n >0:
        c1[n] = (1-m[n])/(1-m[n-1])*(delta[n-1]/delta[n])*alpha[n]**0.5 + (1-m[n-1])*(delta_cond[n]/delta[n])/alpha[n]**0.5
        c2[n] = (m[n-1] * delta[n] - (m[n] *(1-m[n]))/(1-m[n-1])*alpha[n]*delta[n-1])*(alpha_cum[n-1]**0.5/delta[n])
        c3[n] = (1-m[n-1])*(delta_cond[n]/delta[n])*(1-alpha_cum[n])**0.5/(alpha[n])**0.5
      else:
        c1[n] = 1 / alpha[n]**0.5
        c3[n] = c1[n] * beta[n] / (1 - alpha_cum[n])**0.5
    return alpha, beta, alpha_cum,sigmas, c1, c2, c3, delta, delta_bar,T
      

def predict(spectrogram, model,initial_predictor, noisy_signal, alpha, beta, alpha_cum, sigmas,c1, c2, c3, delta, delta_bar,T, device=torch.device('cuda')):
  with torch.no_grad():
    # Expand rank 2 tensors by adding a batch dimension.
    if len(spectrogram.shape) == 2:
      spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)
    
                       # B                          the num of frames     hop_length?           
    audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    #print('noise_scale:{}'.format(noise_scale))
    noisy_audio = torch.zeros(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    noisy_audio[:,:noisy_signal.shape[0]] = torch.from_numpy(noisy_signal).to(device)
    #audio = noisy_audio
    gamma = [0.2]
    for n in range(len(alpha) - 1, -1, -1):
      if n == len(alpha)-1:
        y_init =initial_predictor(noisy_audio).squeeze(1)
        noisy_audio = noisy_audio - y_init
        audio = audio - y_init
      if n > 0:                   
        predicted_noise =  model(audio, noisy_audio,noise_scale[n]).squeeze(1)
        audio = c1[n] * audio + c2[n] * noisy_audio - c3[n] * predicted_noise
        noise = torch.randn_like(audio)
        newsigma= delta_bar[n]**0.5 
        audio += newsigma * noise
      else:
        predicted_noise =  model(audio, noisy_audio, noise_scale[n]).squeeze(1)
        audio = c1[n] * audio - c3[n] * predicted_noise
        audio = audio + y_init
        audio = (1-gamma[n])*audio+gamma[n]*noisy_audio
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate,y_init


def write_waveform(samples,clean,noisy,y_init,sr,writer,step): 
    time = np.arange(0, len(samples)) * (1.0 / sr)
    figure,ax = plt.subplots(nrows=1,ncols=7,figsize=(32, 4))
    ax[0].plot(time,noisy,'-',linewidth=0.5)
    ax[0].set_title('noisy waveform')
    ax[0].set_xlabel('Time[s]')
    ax[1].plot(time,y_init,'-',linewidth=0.5)
    ax[1].set_title('deterministic output waveform')
    ax[1].set_xlabel('Time[s]')
    ax[2].plot(time,clean,'-',linewidth=0.5)
    ax[2].set_title('clean waveform')
    ax[2].set_xlabel('Time[s]')
    ax[3].plot(time,clean-y_init,'-',linewidth=0.5)
    ax[3].set_title('clean-yinit waveform')
    ax[3].set_xlabel('Time[s]')
    ax[4].plot(time,noisy-y_init,'-',linewidth=0.5)
    ax[4].set_title('noisy-y_init waveform')
    ax[4].set_xlabel('Time[s]')
    ax[5].plot(time,(clean-y_init)/2+(noisy-y_init)/2,'-',linewidth=0.5)
    ax[5].set_title('mixture waveform')
    ax[5].set_xlabel('Time[s]')
    gaussian= [random.gauss((noisy-y_init)[i],0.05) for i in range(len(samples))]
    ax[6].plot(time,gaussian,'-',linewidth=0.5)
    ax[6].set_title(f'Gaussian waveform')
    ax[6].set_xlabel('Time[s]')
    writer.add_figure('waveform',figure,step)
def main(args):
  writer = SummaryWriter(os.path.dirname(args.model_dir))
  if args.voicebank:
    test_id_list = parse_csv('/home/qzb521/qinghua/learning-audio-visual-dereverberation-main/CDiffuSE/database/VoiceBank-demand/16k/testset_txt/test.csv')
  else:
    # wsj0 test set
    test_id_list = {}
    file_list = glob('/home/qzb521/qinghua/learning-audio-visual-dereverberation-main/CDiffuSE/wsj0_test_data/wsj0_test_clean/*.wav', recursive=False)
    ids= [os.path.basename(file).split('.wav')[0] for file in file_list]
    idx = 0 
    while idx < len(ids):
        test_id_list[ids[idx]] = file_list[idx]
        idx += 1
  
  if args.eval_asr:
    print('eval_asr')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                                savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                                run_opts={"device":"cuda:0"})
    cer_stats = ErrorRateStats()
    print('loaded csv file, conclude {} files'.format(len(test_id_list)))
  if args.eval_enhancement:
    metrics = speechmetrics.load(['stoi'])
  if args.se:
    base_params.n_mels = 513
  else:
    base_params.n_mels = 80 
  specnames = []
  running_metrics = defaultdict(list)
  for path in args.spectrogram_path:
    specnames += glob(f'{path}/*.wav.spec.npy', recursive=True)
  ################################### load model ####################################
  model = load_model(model_dir=args.model_dir,args=args)
  initial_predictor = load_initial_predictor(model_dir=args.model_dir,args=args)
  model_num_params = sum(p.numel() for p in model.parameters())
  print('the number of model params: {}'.format(model_num_params))
  initial_predictor_num_params = sum(p.numel() for p in initial_predictor.parameters())
  print('the number of initial_predictor params: {}'.format(initial_predictor_num_params))
  print('total parameter:{}'.format(initial_predictor_num_params+model_num_params))
  ######################################################################################
  alpha, beta, alpha_cum, sigmas,c1, c2, c3, delta, delta_bar,T = inference_schedule(model)
  output_path = os.path.join(args.output, specnames[0].split("/")[-2])
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  step = 0
  for spec in tqdm(specnames):
    spectrogram = torch.from_numpy(np.load(spec))
    # 通过语谱图获取音频文件名
    speech_id = spec.split("/")[-1].replace(".wav.spec.npy","")
    if args.voicebank:
      noisy_signal, _ = librosa.load(os.path.join(args.wav_path,spec.split("/")[-1].replace(".spec.npy","")),sr=16000)
      clean_signal, _ = librosa.load(test_id_list[speech_id][1],sr=16000)
    else:
      print(os.path.join(args.wav_path,spec.split("/")[-1].replace(".spec.npy","")))
      print(test_id_list[speech_id])
      noisy_signal, _ = librosa.load(os.path.join(args.wav_path,spec.split("/")[-1].replace(".spec.npy","")),sr=16000)
      clean_signal, _ = librosa.load(test_id_list[speech_id],sr=16000)

    wlen = noisy_signal.shape[0]
    if args.eval_avg:
      avg_num = 1
      audios  = []
      for i in range(avg_num):
        audio,sr,y_init = predict(spectrogram, model,initial_predictor, noisy_signal, alpha, beta, alpha_cum, sigmas,c1, c2, c3, delta, delta_bar,T)
        audio = audio[:,:wlen]
        y_init = y_init[:,:wlen]
        audios.append(audio.cpu()[0].numpy())
      audios = np.array(audios)
      audio = np.sum(audios,axis=0)/avg_num  
      write_waveform(audio,clean_signal,noisy_signal,y_init.cpu()[0].numpy(),sr,writer,step)
    if args.eval_enhancement:
      reference = clean_signal
      enhanced = noisy_signal
      pesq_score = pesq(16000, reference, enhanced, 'wb')
      res = eval_composite(reference, enhanced)
      print('{} pesq:{},csig:{},cbak:{},covl:{}'.format(step,pesq_score,res['csig'],res['cbak'],res['covl']))
      running_metrics['pesq'].append(pesq_score)
      running_metrics['csig'].append(res['csig'])
      running_metrics['cbak'].append(res['cbak'])
      running_metrics['covl'].append(res['covl'])

    if args.eval_asr:
      pred, tokens = asr_model.transcribe_batch(audio, torch.tensor([1.0]))
      _, _, target,_ = test_id_list[speech_id]
      cer_stats.append(ids=[speech_id], predict=np.array([pred[0].split(' ')]),
                             target=np.array([target.split(' ')]))
      print(pred[0].split(' '))
      print(target.upper().split(' '))
    step += 1
    
  if args.eval_enhancement:
    for metric, values in running_metrics.items():
            avg_metric_value = np.mean(values)
            print(metric, avg_metric_value)

  if args.eval_asr:
        wer = cer_stats.summarize()['WER']
        print(f"Final WER:", wer)



if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('spectrogram_path', nargs='+',
      help='space separated list of directories from spectrogram file generated by diffwave.preprocess')
  parser.add_argument('wav_path',
      help='input noisy wav directory')
  parser.add_argument('--output', '-o', default='output/',
      help='output path name')
  parser.add_argument('--fast', dest='fast', action='store_true',
      help='fast sampling procedure')
  parser.add_argument('--full', dest='fast', action='store_false',
      help='fast sampling procedure')
  parser.add_argument('--se', dest='se', action='store_true')
  parser.add_argument('--voicebank', dest='voicebank', action='store_true')
  parser.add_argument('--wsj0', dest='wsj0', action='store_true')
  parser.add_argument('--eval_asr',default=False,action='store_true')
  parser.add_argument('--eval_enhancement',default=True, action='store_true')
  parser.add_argument('--eval_avg',default=False)
  parser.set_defaults(se=True)
  parser.set_defaults(fast=True)
  main(parser.parse_args())
