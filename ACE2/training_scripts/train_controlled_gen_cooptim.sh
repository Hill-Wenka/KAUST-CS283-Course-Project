#!bin/sh

# 1 A100 80 GB: largest training batch size is 30 (use cyc_cons) for 6 layers, eval 256
# 1 A100 80 GB: largest training batch size is 10 (use cyc_cons) for 24 layers, eval 128
# 8 A100 40 GB: largest training batch size is 80 (use cyc_cons)
# 4 A100 40 GB: largest training batch size is 32 (use cyc_cons)
# 2 A100 40 GB: largest training batch size is 20 (use cyc_cons)

#--lambda_contrastive_perturb_cyc=1.0 \
#--contrastive_perturb_cyc_start_step 10000 \

# ddG
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./train_controlled_generator_cooptim.py \
#  --lr=1e-04 \
#  --num_train_epochs=30 \
#  --train_ratio=0.9 \
#  --lambda_contrastive=1.0 \
#  --latent_pooler=cls \
#  --pool_enc_hidden_states_for_dec \
#  --latent_space_type=wae \
#  --beta_start_step 50000 \
#  --latent_size=1024 \
#  --wae_z_enc_type=deterministic \
#  --no_separate_latent_enc \
#  --no_separate_latent_dec \
#  --lambda_contrastive_cyc=1.0 \
#  --contrastive_cyc_start_step 50000 \
#  --eval_steps 500 \
#  --save_steps 1000 \
#  --logging_steps 20 \
#  --per_device_train_batch_size 32 \
#  --per_device_eval_batch_size 128 \
#  --data_dir=/home/hew/storage/storage/genhance/data/ \
#  --pretrained_dir=/home/hew/storage/storage/genhance/pretrained/ \
#  --cache_dir=/home/hew/storage/storage/genhance/pretrained/ \
#  --src_json=/home/hew/python/genhance/temp/config.json \
#  --output_dir=/home/hew/storage/storage/genhance/ckpts/congen8_new_ddG_6_layer/results/ \
#  --logging_dir=/home/hew/python/genhance/tensorboard/congen8_new_ddG_6_layer \
#  --property ddG \
#  --num_layers 6 \
#  --num_decoder_layers 6 \
#  --cuda_device 0,1 \
#  --datafile train_new_tophalf_ddG.pkl

# solubility
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./train_controlled_generator_cooptim.py \
#  --lr=1e-04 \
#  --num_train_epochs=30 \
#  --train_ratio=0.9 \
#  --lambda_contrastive=1.0 \
#  --latent_pooler=cls \
#  --pool_enc_hidden_states_for_dec \
#  --latent_space_type=wae \
#  --beta_start_step 50000 \
#  --latent_size=1024 \
#  --wae_z_enc_type=deterministic \
#  --no_separate_latent_enc \
#  --no_separate_latent_dec \
#  --lambda_contrastive_cyc=1.0 \
#  --contrastive_cyc_start_step 50000 \
#  --eval_steps 500 \
#  --save_steps 1000 \
#  --logging_steps 20 \
#  --per_device_train_batch_size 32 \
#  --per_device_eval_batch_size 128 \
#  --data_dir=/home/hew/storage/storage/genhance/data/ \
#  --pretrained_dir=/home/hew/storage/storage/genhance/pretrained/ \
#  --cache_dir=/home/hew/storage/storage/genhance/pretrained/ \
#  --src_json=/home/hew/python/genhance/temp/config.json \
#  --output_dir=/home/hew/storage/storage/genhance/ckpts/congen9_new_solubility_6_layer/results/ \
#  --logging_dir=/home/hew/python/genhance/tensorboard/congen9_new_solubility_6_layer \
#  --property solubility \
#  --num_layers 6 \
#  --num_decoder_layers 6 \
#  --cuda_device 0,1 \
#  --datafile train_new_tophalf_solubility.pkl

# ddG_solubility
sudo /home/hew/miniconda3/envs/genhance/bin/python ./train_controlled_generator_cooptim.py \
  --lr=1e-04 \
  --num_train_epochs=30 \
  --train_ratio=0.9 \
  --lambda_contrastive=1.0 \
  --latent_pooler=cls \
  --pool_enc_hidden_states_for_dec \
  --latent_space_type=wae \
  --beta_start_step 50000 \
  --latent_size=1024 \
  --wae_z_enc_type=deterministic \
  --no_separate_latent_enc \
  --no_separate_latent_dec \
  --lambda_contrastive_cyc=1.0 \
  --contrastive_cyc_start_step 50000 \
  --eval_steps 500 \
  --save_steps 1000 \
  --logging_steps 20 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --data_dir=/home/hew/storage/storage/genhance/data/ \
  --pretrained_dir=/home/hew/storage/storage/genhance/pretrained/ \
  --cache_dir=/home/hew/storage/storage/genhance/pretrained/ \
  --src_json=/home/hew/python/genhance/temp/config.json \
  --output_dir=/home/hew/storage/storage/genhance/ckpts/congen10_new_ddG_solubility_6_layer/results/ \
  --logging_dir=/home/hew/python/genhance/tensorboard/congen10_new_ddG_solubility_6_layer \
  --property ddG_solubility \
  --num_layers 6 \
  --num_decoder_layers 6 \
  --cuda_device 0,1 \
  --datafile train_new_tophalf_avg_rank.pkl
