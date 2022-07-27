import torch
import torch.nn as nn
import torch.functional as F


class Encoder(nn.Module):
  def __init__(self, input_dim, latent_dim, arch):
    super().__init__()

    layers = []
    for l in arch:
      layers.append(nn.Linear(input_dim, l))
      layers.append(nn.ReLU())
      input_dim = l

    self.fcs = nn.Sequential(*layers)
    self.last_fc = nn.Linear(input_dim, latent_dim)

  def forward(self, inputs):
    h = self.fcs(inputs)
    return self.last_fc(h)

class Decoder(nn.Module):
  def __init__(self, latent_dim, input_dim, arch):
    super().__init__()

    layers = []
    for l in arch:
      layers.append(nn.Linear(latent_dim, l))
      layers.append(nn.ReLU())
      latent_dim = l

    self.fcs = nn.Sequential(*layers)
    self.last_fc = nn.Linear(latent_dim, input_dim)
  
  def forward(self, inputs):
    h = self.fcs(inputs)
    return self.last_fc(h)

class AutoEncoder(nn.Module):
  def __init__(self, input_dim, latent_dim, arch_enc, arch_dec):
    super().__init__()
    
    self.encoder = Encoder(input_dim, latent_dim, arch_enc)
    self.decoder = Decoder(latent_dim, input_dim, arch_dec)
  
  def forward(self, inputs):
    latent = self.encoder(inputs)
    return self.decoder(latent)