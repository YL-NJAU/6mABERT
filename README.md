# Deep6mA

6mABERT is a deep-learning-based framework to predict 6mA-containing sequences. 

# Dependency

- Python3
- tensorflow==1.12.0


# Content

- origion_data:data of rice for model training ( data of Arabidopsis thaliana, Fragaria vesca and Rosa chinensis for validation on other three species)
- data: 
  - train1:
	- train.tsv
	- dev.tsv
	- test.tsv
  - train2
  	- train.tsv
	- dev.tsv
	- test.tsv
  - train3
    ...
   #train:test:dev = 8:1:1 (The data of rice were randomly allocated according to 8:1:1)
   
- pre_models: 
  - uncased_L-12_H-768_A-12 
  #download from (https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) 
  
- result: 
  - outputs: trained models for rice(10-fold)
  
    

# Usage

bash run_6mA.sh
	
# What's in run_6mA.sh

'''
for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=2 \
	python run_abs.py \
	 --task_name=6ma_classify \
	 --do_train=true \ 
	 --do_eval=true \
	 --do_predict=False \
	 --data_dir=data/train$i \
	 --vocab_file=pre_models/uncased_L-12_H-768_A-12/vocab.txt \
	 --bert_config_file=pre_models/uncased_L-12_H-768_A-12/bert_config.json \
	 --init_checkpoint=pre_models/uncased_L-12_H-768_A-12/bert_model.ckpt \
	 --max_seq_length=128 \
	 --train_batch_size=32 \
	 --learning_rate=2e-5 \
	 --num_train_epochs=3.0 \
	 --output_dir=./outputs/output_$i/

done
'''

This script output the trained model and prediction result in the outputs. 

