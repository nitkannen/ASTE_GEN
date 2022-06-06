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
                --use_tagger True \
                --logger_name 14res_hypsearch_4_4_1e_noRegressor.txt \
                --log_message hyp_4_4_noregressor \
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
                --beta 0.1 \
                --use_tagger True \
                --logger_name 14res_hypsearch_4_4_1e_beta0.1.txt \
                --log_message hyp_4_4_beta0.1 \
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
                --logger_name 14res_hypsearch_4_4_1e_notagger.txt \
                --log_message hyp_4_4_notagger \
                --gpu_id 1




