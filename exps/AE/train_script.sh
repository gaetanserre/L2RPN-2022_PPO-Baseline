set -x

python train.py \
--nb_epochs 50 \
--latent_dim 400 \
--arch_enc 1024 800 600 \
--arch_dec 600 800 1024 \
--batch_size 512 \
--nb_models 10