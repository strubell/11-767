export GLUE_DIR=./glue_data/
export TASK_NAME=QQP

CUDA_VISIBLE_DEVICES=0 python ./run_glue_with_pabee.py \
  --model_type distilbert \
  --model_name_or_path distilbert-base-uncased \
  --task_name $TASK_NAME \
  --do_lower_case \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --save_steps 2000 \
  --logging_steps 2000 \
  --num_train_epochs 1 \
  --output_dir ./output/ \
  --evaluate_during_training \
  --inference_time_by_layer
