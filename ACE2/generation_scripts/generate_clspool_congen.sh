#!bin/sh

#  8 A100 40 GB: largest generation batch size is 800
#  1 A100 80 GB: largest generation batch size is 1200

# congen2
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
#  --gen_save_interval=10 \
#  --num_generations=260000 \
#  --topk_as_input=12500 \
#  --gen_batch_size=1200 \
#  --num_gen_samples_per_input=20 \
#  --unique_gen \
#  --z_tar_edit_before_dec=-2.0 \
#  --temperature_init=1.0 \
#  --temperature_multiple=1.004 \
#  --input_data_dir=/home/hew/storage/storage/genhance/data \
#  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen2/results/step_42000 \
#  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen2/generations/ \
#  --prepend_output_name=step42000_edit_more_2 \
#  --property ddG \
#  --cuda_device 2

# congen3
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
#  --gen_save_interval=10 \
#  --num_generations=250000 \
#  --topk_as_input=12500 \
#  --gen_batch_size=1200 \
#  --num_gen_samples_per_input=20 \
#  --unique_gen \
#  --z_tar_edit_before_dec=0.92 \
#  --temperature_init=1.0 \
#  --temperature_multiple=1.004 \
#  --input_data_dir=/home/hew/storage/storage/genhance/data \
#  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen3/results/step_42000 \
#  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen3/generations/ \
#  --prepend_output_name=step42000 \
#  --property solubility \
#  --cuda_device 1

# congen4
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
#  --gen_save_interval=10 \
#  --num_generations=250000 \
#  --topk_as_input=12500 \
#  --gen_batch_size=1200 \
#  --num_gen_samples_per_input=20 \
#  --unique_gen \
#  --z_tar_edit_before_dec -1.0 0.92 \
#  --temperature_init=1.0 \
#  --temperature_multiple=1.001 \
#  --input_data_dir=/home/hew/storage/storage/genhance/data \
#  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen4/results/step_42000 \
#  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen4/generations/ \
#  --prepend_output_name=step42000 \
#  --property ddG_solubility \
#  --cuda_device 2

# congen2 continue
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
#  --gen_save_interval=10 \
#  --num_generations=250000 \
#  --topk_as_input=12500 \
#  --gen_batch_size=1200 \
#  --num_gen_samples_per_input=20 \
#  --unique_gen \
#  --z_tar_edit_before_dec=-1.1 \
#  --temperature_init=2.0 \
#  --temperature_multiple=1.004 \
#  --input_data_dir=/home/hew/storage/storage/genhance/data \
#  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen2/results/step_42000 \
#  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen2/generations/ \
#  --prepend_output_name=step42000 \
#  --property ddG \
#  --cuda_device 0 \
#  --skip_gen

# congen3 continue
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
#  --gen_save_interval=10 \
#  --num_generations=250000 \
#  --topk_as_input=12500 \
#  --gen_batch_size=1200 \
#  --num_gen_samples_per_input=20 \
#  --unique_gen \
#  --z_tar_edit_before_dec=0.92 \
#  --temperature_init=1.5 \
#  --temperature_multiple=1.004 \
#  --input_data_dir=/home/hew/storage/storage/genhance/data \
#  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen3/results/step_42000 \
#  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen3/generations/ \
#  --prepend_output_name=step42000 \
#  --property solubility \
#  --cuda_device 1 \
#  --skip_gen

# congen4 continue
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
#  --gen_save_interval=10 \
#  --num_generations=250000 \
#  --topk_as_input=12500 \
#  --gen_batch_size=1200 \
#  --num_gen_samples_per_input=20 \
#  --unique_gen \
#  --z_tar_edit_before_dec -1.0 0.92 \
#  --temperature_init=1.0 \
#  --temperature_multiple=1.001 \
#  --input_data_dir=/home/hew/storage/storage/genhance/data \
#  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen4/results/step_42000 \
#  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen4/generations/ \
#  --prepend_output_name=step42000 \
#  --property ddG_solubility \
#  --cuda_device 2 \
#  --skip_gen

# congen8
# gen_batch_size=1200
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
#  --gen_save_interval=10 \
#  --num_generations=260000 \
#  --topk_as_input=12500 \
#  --gen_batch_size=600 \
#  --num_gen_samples_per_input=20 \
#  --unique_gen \
#  --z_tar_edit_before_dec=-40.0 \
#  --temperature_init=1.0 \
#  --temperature_multiple=1.001 \
#  --input_data_dir=/home/hew/storage/storage/genhance/data \
#  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen8_new_ddG_6_layer/results/ \
#  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen8_new_ddG_6_layer/generations_[-40]_[1.0]/ \
#  --prepend_output_name=congen8_new_ddG_6_layer \
#  --property ddG \
#  --cuda_device 0,1 \
#  --datafile train_new_tophalf_ddG.pkl

# congen9
#sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
#  --gen_save_interval=10 \
#  --num_generations=260000 \
#  --topk_as_input=12500 \
#  --gen_batch_size=600 \
#  --num_gen_samples_per_input=20 \
#  --unique_gen \
#  --z_tar_edit_before_dec=1.0 \
#  --temperature_init=1.0 \
#  --temperature_multiple=1.001 \
#  --input_data_dir=/home/hew/storage/storage/genhance/data \
#  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen9_new_solubility_6_layer/results/ \
#  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen9_new_solubility_6_layer/generations_[1.0]_[1.0]/ \
#  --prepend_output_name=congen9_new_solubility_6_layer \
#  --property solubility \
#  --cuda_device 0,1 \
#  --datafile train_new_tophalf_solubility.pkl


# congen10
sudo /home/hew/miniconda3/envs/genhance/bin/python ./generate_sequences_congen.py \
  --gen_save_interval=10 \
  --num_generations=260000 \
  --topk_as_input=12500 \
  --gen_batch_size=400 \
  --num_gen_samples_per_input=20 \
  --unique_gen \
  --z_tar_edit_before_dec -40.0 1.0 \
  --temperature_init=1.0 \
  --temperature_multiple=1.001 \
  --input_data_dir=/home/hew/storage/storage/genhance/data \
  --gen_pretrained_dir=/home/hew/storage/storage/genhance/ckpts/congen10_new_ddG_solubility_6_layer/results/ \
  --generation_output_dir=/home/hew/storage/storage/genhance/ckpts/congen10_new_ddG_solubility_6_layer/generations_[-40.0]_[1.0]_[1.0]/ \
  --prepend_output_name=congen10_new_ddG_solubility_6_layer \
  --property ddG_solubility \
  --cuda_device 0,1 \
  --datafile train_new_tophalf_avg_rank.pkl