#!bin/sh

# 8 A100 40 GB: largest training batch size is 60 (use cyc_cons)
# 4 A100 40 GB: largest training batch size is 32 (use cyc_cons)
# 2 A100 40 GB: largest training batch size is 20 (use cyc_cons)

#  --per_device_train_batch_size 60 \
#  --num_train_epochs=20 \
sudo /home/hew/miniconda3/envs/genhance/bin/python ./train_controlled_generator_continue.py \
  --pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen6_ddG_24_layer/results/step_36000 \
  --cuda_device 0,1
