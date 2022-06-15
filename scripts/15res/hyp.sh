python main.py --task 15res  --train_dataset_path 15res/train  --dev_dataset_path 15res/dev  --test_dataset_path 15res/test  --model_name_or_path t5-base  --do_train   --do_eval  --train_batch_size 2  --gradient_accumulation_steps 2 --eval_batch_size 16  --learning_rate 3e-4  --num_train_epochs 20  --regressor True  --use_tagger True --model_weights models/model_after6epochs  --logger_name 15res_logs_regressor0.2_and_tagger_with_contrast6.txt  --log_message epoch6_2_2_3e4_0.2default --gpu_id 1

python main.py --task 15res  --train_dataset_path 15res/train  --dev_dataset_path 15res/dev  --test_dataset_path 15res/test  --model_name_or_path t5-base  --do_train   --do_eval  --train_batch_size 2  --gradient_accumulation_steps 2 --eval_batch_size 16  --learning_rate 3e-4  --num_train_epochs 20  --regressor True  --use_tagger True --beta 0.4 --model_weights models/model_after6epochs  --logger_name 15res_logs_regressor0.4_and_tagger_with_contrast6.txt  --log_message epoch6_2_2_3e4_0.4default --gpu_id 1

python main.py --task 15res  --train_dataset_path 15res/train  --dev_dataset_path 15res/dev  --test_dataset_path 15res/test  --model_name_or_path t5-base  --do_train   --do_eval  --train_batch_size 4  --gradient_accumulation_steps 4 --eval_batch_size 16  --learning_rate 3e-4  --num_train_epochs 20  --regressor True  --use_tagger True --beta 0.4 --model_weights models/model_after6epochs  --logger_name 15res_logs_regressor0.4_and_tagger_with_contrast6.txt  --log_message epoch6_4_4_3e4_0.4default --gpu_id 1

python main.py --task 15res  --train_dataset_path 15res/train  --dev_dataset_path 15res/dev  --test_dataset_path 15res/test  --model_name_or_path t5-base  --do_train   --do_eval  --train_batch_size 4  --gradient_accumulation_steps 2  --eval_batch_size 16  --learning_rate 3e-4  --num_train_epochs 20  --regressor True  --use_tagger True --beta 0.4 --model_weights models/model_after6epochs  --logger_name 15res_logs_regressor0.4_and_tagger_with_contrast6.txt  --log_message epoch6_4_2_3e4_0.4default --gpu_id 1