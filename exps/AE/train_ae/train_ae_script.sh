set -x

python train_ae.py \
--dataset dataset_254397obs.npz \
--nb_epochs 5 \
--latent_dim 400 \
--arch_enc 1024 800 600 \
--arch_dec 600 800 1024 \
--batch_size 512 \
--nb_models 3