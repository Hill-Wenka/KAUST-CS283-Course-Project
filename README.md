# KAUST-CS283-Course-Project
The course project of CS283 Deep Generative Modeling at KAUST. The modification of GENhance for thermostability and solubility co-optimization.

# How to run

## Dependency
1. The yaml file for the anaconda environment is stored on ```./genhance/temp/genhance.yaml```. Create the anaconda virtual encironemtn and install dependencies: ```conda env create -f ./genhance/temp/genhance.yaml```.
2. Install **FoldX 5.0** and **Protein-Sol**.

## Training
1. You are required more than 40 GB GPU memory to run as I trained the model using at least 1 A100-80GB or 2 A100-40GB and the memory size depends on the used layer number of the encoder and decoder (6 + 6 by default). Larger model usually gives better performance under the same setting.
2. All related preprossed data are stored in ```./genhance/data/```.
3. Make sure you download the ProtT5 model from https://huggingface.co/Rostlab/prot_t5_xl_uniref50.
4. Modify all path parameters in the script file '''./genhance/ACE2/training_scripts/train_controlled_gen_cooptim.sh''', for example, Set the ```pretrained_dir``` and ```cache_dir``` in the script file  with your T5 model path. 
5. Run the following commands in shell: ```nohup sh ./genhance/ACE2/training_scripts/train_controlled_gen_cooptim.sh > ./run.log 2>&1 &```
7. The default setting is for the thermostability and solubility co-optimization. Also, you can train the model for thermostability or solubility optimization and modify the configuration by commenting or uncommenting the corresponding scripts in ```./genhance/ACE2/training_scripts/train_controlled_gen_cooptim.sh```.
8. The training takes up approximately 24 hours on A100.


## Generation
1. Make sure you run this step after the training finished.
2. Modify all path parameters in the script file '''./genhance/ACE2/generation_scripts/generate_clspool_congen.sh''', for example, Set the ```gen_pretrained_dir``` with your path storing the checkpoint model. 
3. Run the following commands in shell: ```sh /home/hew/python/genhance/ACE2/generation_scripts/generate_clspool_congen.sh```.
4. The default setting is for the thermostability and solubility co-optimization. Also, you can train the model for thermostability or solubility optimization and modify the configuration by commenting or uncommenting the corresponding scripts in ```./genhance/ACE2/generation_scripts/generate_clspool_congen.sh```
5. The generation process takes up approximately 10 hours on A100.


## Prepare Sequences for FoldX and Protein-Sol
1. Open the jupyter notebook ```./genhance/ACE2/Prepare Sequences for FoldX.ipynb```
2. Modity the corresponding variables, for example, generation path, and follow the instruction to filter out those sequences can not meet the requirements.


## Evaluation
1. The evaluation jupyter notebooks are stored in ```./genhance/ACE2/evaluation/```
2. Open the corresponding jupyter notebooks, modify the relevant paths, for example, the excutable path of FoldX 5.0 and follow the instructions.










