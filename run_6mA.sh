#!bin/sh/
for i in 1 2 3 4 5 6 7 8 9 10
do

CUDA_VISIBLE_DEVICES=2 python run_6mA.py --task_name=6ma_classify --do_train=true --do_eval=true --do_predict=true --data_dir=data/train$i --vocab_file=pre_models/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=pre_models/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=pre_models/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./outputs/output_$i/

done