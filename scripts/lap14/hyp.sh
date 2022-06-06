python main.py --task lap14 \
                --train_dataset_path lap14/train \
                --dev_dataset_path lap14/dev \
                --test_dataset_path lap14/test \
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
                --logger_name lap14_logs_hyp_search2_2.txt \
                --log_message hyp_search \
                --gpu_id 1 

python main.py --task lap14 \
                --train_dataset_path lap14/train \
                --dev_dataset_path lap14/dev \
                --test_dataset_path lap14/test \
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
                --logger_name lap14_logs_hyp_search4_2.txt \
                --log_message hyp_search \
                --gpu_id 1 


python main.py --task lap14 \
                --train_dataset_path lap14/train \
                --dev_dataset_path lap14/dev \
                --test_dataset_path lap14/test \
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
                --logger_name lap14_logs_hyp_search_4_4.txt \
                --log_message hyp_search \
                --gpu_id 1 

python main.py --task lap14 \
                --train_dataset_path lap14/train \
                --dev_dataset_path lap14/dev \
                --test_dataset_path lap14/test \
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
                --logger_name lap14_logs_hyp_search_4_4_1e.txt \
                --log_message hyp_search \
                --gpu_id 1 

python main.py --task lap14 \
                --train_dataset_path lap14/train \
                --dev_dataset_path lap14/dev \
                --test_dataset_path lap14/test \
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
                --logger_name lap14_logs_hyp_search_4_4_5e.txt \
                --log_message hyp_search \
                --gpu_id 1 