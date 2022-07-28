set -x

python train_ae.py \
--dataset dataset_516252obs.npz \
--nb_epochs 60 \
--latent_dim 400 \
--arch_enc 1024 800 600 \
--arch_dec 600 800 1024 \
--batch_size 1024 \
--nb_models 10