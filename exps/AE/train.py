import argparse

from ae import AutoEncoder

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def create_obs_dataset(filename):
  observations = np.load(filename)["observations"]
  obs          = TensorDataset(torch.from_numpy(observations))
  lengths      = [int(i*len(observations)) for i in [0.7, 0.2]]
  lengths.append(len(observations) - (lengths[0] + lengths[1]))
  return random_split(obs, lengths)

def plot_learning_curve(losses, y_label, percentiles, path=None):
  x_plt   = range(losses.shape[1])
  x_displ = [el + 1 for el in x_plt]

  median_losses = np.median(losses, axis=0)
  max_y_axis    = max(median_losses) + 0.05
  min_y_axis    = min(median_losses) - 0.05

  plt.figure(figsize=(10, 6))
  plt.fill_between(x_displ,
                  y1=np.percentile(losses, percentiles[0], axis=0),
                  y2=np.percentile(losses, percentiles[3], axis=0),
                  color="cornflowerblue",
                  alpha=0.3,
                  label=f"[{percentiles[0]}%-{percentiles[-1]}%]"
                  )
  plt.fill_between(x_displ,
                  y1=np.percentile(losses, percentiles[1], axis=0),
                  y2=np.percentile(losses, percentiles[2], axis=0),
                  color="cornflowerblue",
                  alpha=0.7,
                  label=f"[{percentiles[1]}%-{percentiles[-2]}%]"
                  )
  plt.plot(x_displ,
          median_losses, 
          color="cornflowerblue",
          label="median")

  plt.ylim(min_y_axis, max_y_axis)
  plt.xlabel("Epoch", fontsize=15)
  plt.ylabel(y_label, fontsize=15)
  plt.grid()
  plt.legend()

  if path is not None:
    plt.savefig(path)
  else:
    plt.show()

def get_test_loss(ae, loss, obs, batch_size):
  ae.eval()

  dataloader = DataLoader(obs, batch_size=batch_size, shuffle=True, pin_memory=True)
  avg_loss = 0
  nb_batch = 0
  for data in dataloader:
    nb_batch += 1

    data      = data[0].to(device)
    data_pred = ae(data)

    l         = loss(data_pred, data)
    avg_loss += l.item()

  ae.train()
  return avg_loss / nb_batch

def train_autoencoder(ae,
                      loss,
                      optimizer,
                      train_obs,
                      valid_obs,
                      test_obs,
                      nb_epochs,
                      batch_size,
                      verbose=False):

  ae.train()
  train_losses = np.zeros(nb_epochs)
  valid_losses = np.zeros(nb_epochs)
  for epoch in range(nb_epochs):
    dataloader = DataLoader(train_obs, batch_size=batch_size, shuffle=True, pin_memory=True)
    if verbose:
      print(f"Epoch {epoch+1}/{nb_epochs}")

    avg_loss = 0
    nb_batch = 0
    for data in dataloader:
      nb_batch += 1

      data      = data[0].to(device)
      data_pred = ae(data)

      l         = loss(data_pred, data)
      avg_loss += l.item()
      l.backward()

      nn.utils.clip_grad_value_(ae.parameters(), 5.0)
      optimizer.step()

    train_losses[epoch] = avg_loss / nb_batch
    val_loss            = get_test_loss(ae, loss, valid_obs, batch_size)
    valid_losses[epoch] = val_loss
    if verbose:
      print(f"Train loss {avg_loss / nb_batch:.4f}")
      print(f"Valid loss {val_loss:.4f}")

  test_loss = get_test_loss(ae, loss, test_obs, batch_size)
  if verbose:
    print(f"Test loss {test_loss:.4f}")

  return test_loss, train_losses, valid_losses

def cli():
  parser = argparse.ArgumentParser(description="Train autoencoder")
  parser.add_argument("--nb_epochs", required=True, type=int,
                      help="Number of epochs")
  parser.add_argument("--batch_size", default=64, type=int,
                      help="Batch size")
  parser.add_argument("--latent_dim", default=40, type=int,
                      help="Dimension of the latent space")
  parser.add_argument("--arch_enc", nargs="+", type=int,
                      default=[],
                      help="Architecture of hidden layers for the encoder")
  parser.add_argument("--arch_dec", nargs="+", type=int,
                      default=[],
                      help="Architecture of hidden layers for the decoder")
  parser.add_argument("--nb_models", default=10, type=int,
                      help="Number of model instances to run")
  parser.add_argument("--output_weights", default="ae_weights", type=str,
                      help="Where to save the autoencoder's weights")
  
  return parser.parse_args()

if __name__ == "__main__":
  args = cli()

  train_losses = np.zeros((args.nb_models, args.nb_epochs))
  val_losses   = np.zeros((args.nb_models, args.nb_epochs))
  test_losses  = np.zeros(args.nb_models)
  percentiles  = [20, 40, 60, 80]

  for i in tqdm(range(args.nb_models)):
    train_obs, valid_obs, test_obs = create_obs_dataset("dataset_254397obs.npz")

    obs_shape = train_obs[0][0].shape[0]
    ae        = AutoEncoder(obs_shape, args.latent_dim, args.arch_enc, args.arch_dec).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), weight_decay=1e-4)

    loss = torch.nn.L1Loss()

    test_losses[i], train_loss, val_loss = train_autoencoder(ae,
                                                          loss,
                                                          optimizer,
                                                          train_obs,
                                                          valid_obs,
                                                          test_obs,
                                                          args.nb_epochs,
                                                          args.batch_size)
    
    train_losses[i] = train_loss
    val_losses[i]   = val_loss

  plot_learning_curve(train_losses, "Train loss", percentiles, path="train_loss.pdf")
  plot_learning_curve(val_losses, "Valid loss", percentiles, path="val_loss.pdf")
  print(f"Test loss mean: {test_losses.mean():.4f}, std: {test_losses.std():.4f} and median: {np.median(test_losses):.4f}")
  torch.save(ae.state_dict(), args.output_weights+".pty")
