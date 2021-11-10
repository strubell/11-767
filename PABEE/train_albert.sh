export GLUE_DIR=./data/
export TASK_NAME=CoLA

CUDA_VISIBLE_DEVICES=0 python ./run_glue_with_pabee.py \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 50 \
  --logging_steps 50 \
  --num_train_epochs 5 \
  --output_dir ./output/ \
  --evaluate_during_training

