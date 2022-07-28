set -x

python train_policy.py \
--ae_weights train_ae/ae_weights.pt \
--latent_dim 400 \
--has_cuda 1 \
--nb_training 1 \
--chronics_name="2050-02-14" \
--agent_name AEMlpPolicy \
--training_iter 1000000