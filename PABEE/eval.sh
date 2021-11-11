export GLUE_DIR=./glue_data
export TASK_NAME=QQP

CUDA_VISIBLE_DEVICES=0 python ./run_glue_with_pabee.py \
  --model_type distilbert \
  --model_name_or_path ./output/ \
  --task_name $TASK_NAME \
  --do_eval \
  --do_lower_case \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --logging_steps 50 \
  --num_train_epochs 15 \
  --output_dir ./output/ \
  --eval_all_checkpoints \
  --patience 3,6 \
  --benchmark
