#train
python train.py --do_train --do_eval --bert_model models/roberta --saving_path output/RoBERTa --learning_rate 4e-5 --num_epoch 10 --batch_size 8 --gradient_accum 4 --fp16 --do_ema --do_adv
#python train.py --do_train --do_eval --bert_model models/wobert --saving_path output/WoBERT --learning_rate 4e-5 --num_epoch 10 --batch_size 32 --fp16 --do_ema --do_adv
#python train.py --do_train --do_eval --bert_model models/wwm --saving_path output/WWM --learning_rate 4e-5 --num_epoch 10 --batch_size 32 --fp16 --do_ema --do_adv
#test
python train.py --do_infer --saving_path output/RoBERTa
#python train.py --do_eval --saving_path output/WoBERT
#python train.py --do_eval --saving_path output/WWM