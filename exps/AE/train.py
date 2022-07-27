from ae import AutoEncoder
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def create_obs_dataset(filename):
  observations = np.load(filename)["observations"]
  obs = TensorDataset(torch.from_numpy(observations))
  lengths = [int(i*len(observations)) for i in [0.7, 0.2]]
  lengths.append(len(observations) - (lengths[0] + lengths[1]))
  return random_split(obs, lengths)

def plot_loss(losses, label, path=None):
  plt.plot(range(len(losses)), losses, label=label)
  plt.xlabel("Epoch")
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

    data = data[0].to(device)
    data_pred = ae(data)

    l = loss(data_pred, data)
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
                      batch_size):

  ae.train()
  train_losses = []
  valid_losses = []
  for epoch in range(nb_epochs):
    dataloader = DataLoader(train_obs, batch_size=batch_size, shuffle=True, pin_memory=True)
    print(f"Epoch {epoch+1}/{nb_epochs}")

    avg_loss = 0
    nb_batch = 0
    for data in dataloader:
      nb_batch += 1

      data = data[0].to(device)
      data_pred = ae(data)

      l = loss(data_pred, data)
      avg_loss += l.item()
      l.backward()
      nn.utils.clip_grad_value_(ae.parameters(), 5.0)
      optimizer.step()

    val_loss = get_test_loss(ae, loss, valid_obs, batch_size)
    valid_losses.append(val_loss)
    train_losses.append(avg_loss / nb_batch)
    print(f"Train loss {avg_loss / nb_batch:.4f}")
    print(f"Valid loss {val_loss:.4f}")

  test_loss = get_test_loss(ae, loss, test_obs, batch_size)
  print(f"Test loss {test_loss:.4f}")
  plot_loss(train_losses, "Train loss")
  plot_loss(valid_losses, "Valid loss")

def cli():
  parser = argparse.ArgumentParser(description="Train auto-encoder")
  parser.add_argument("--nb_epochs", required=True, type=int,
                      help="Number of epochs")
  parser.add_argument("--batch_size", default=64, type=int,
                      help="Batch size")
  parser.add_argument("--latent_dim", default=40, type=int,
                      help="Dimension of the latent space")
  
  return parser.parse_args()

if __name__ == "__main__":
  args = cli()

  train_obs, valid_obs, test_obs = create_obs_dataset("dataset_254397obs.npz")
  obs_shape = train_obs[0][0].shape[0]

  arch_enc = [1024, 800]
  arch_dec = [800, 1024]
  ae = AutoEncoder(obs_shape, args.latent_dim, arch_enc, arch_dec).to(device)
  loss = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(ae.parameters(), weight_decay=1e-4)


  train_autoencoder(ae,
                    loss,
                    optimizer,
                    train_obs,
                    valid_obs,
                    test_obs,
                    args.nb_epochs,
                    args.batch_size)
