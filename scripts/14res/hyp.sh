python main.py --task 14res \
                --train_dataset_path 14res/train \
                --dev_dataset_path 14res/dev \
                --test_dataset_path 14res/test \
                --model_name_or_path t5-base \
                --do_train \
                --do_eval \
                --train_batch_size 2 \
                --gradient_accumulation_steps 2 \
                --eval_batch_size 16 \
                --learning_rate 3e-4 \
                --num_train_epochs 20 \
                --regressor True \
                --use_tagger True \
                --logger_name 14res_hypsearch_2_2.txt \
                --log_message hyp_2_2 \
                --gpu_id 1

python main.py --task 14res \
                --train_dataset_path 14res/train \
                --dev_dataset_path 14res/dev \
                --test_dataset_path 14res/test \
                --model_name_or_path t5-base \
                --do_train \
                --do_eval \
                --train_batch_size 4 \
                --gradient_accumulation_steps 2 \
                --eval_batch_size 16 \
                --learning_rate 3e-4 \
                --num_train_epochs 20 \
                --regressor True \
                --use_tagger True \
                --logger_name 14res_hypsearch_4_2.txt \
                --log_message hyp_2_2 \
                --gpu_id 1

python main.py --task 14res \
                --train_dataset_path 14res/train \
                --dev_dataset_path 14res/dev \
                --test_dataset_path 14res/test \
                --model_name_or_path t5-base \
                --do_train \
                --do_eval \
                --train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --eval_batch_size 16 \
                --learning_rate 3e-4 \
                --num_train_epochs 20 \
                --regressor True \
                --use_tagger True \
                --logger_name 14res_hypsearch_4_4_3e.txt \
                --log_message hyp_2_2 \
                --gpu_id 1

python main.py --task 14res \
                --train_dataset_path 14res/train \
                --dev_dataset_path 14res/dev \
                --test_dataset_path 14res/test \
                --model_name_or_path t5-base \
                --do_train \
                --do_eval \
                --train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --eval_batch_size 16 \
                --learning_rate 1e-4 \
                --num_train_epochs 20 \
                --regressor True \
                --use_tagger True \
                --logger_name 14res_hypsearch_4_4_1e.txt \
                --log_message hyp_2_2 \
                --gpu_id 1

python main.py --task 14res \
                --train_dataset_path 14res/train \
                --dev_dataset_path 14res/dev \
                --test_dataset_path 14res/test \
                --model_name_or_path t5-base \
                --do_train \
                --do_eval \
                --train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --eval_batch_size 16 \
                --learning_rate 5e-4 \
                --num_train_epochs 20 \
                --regressor True \
                --use_tagger True \
                --logger_name 14res_hypsearch_4_4_5e.txt \
                --log_message hyp_2_2 \
                --gpu_id 1