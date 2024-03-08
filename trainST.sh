#!/bin/bash

args=(
  # data arguments
  --data_dir data/kodie
  --label c_vulg
  --labels_num 4
  --max_seq_length 256
  --text_col c_text_en
  --output_dir ./trained\ adapters/trained_adapters_roberta_en/vulgär
  # model arguments
  --model_name_or_path roberta-base
  # training arguments
  --learning_rate 0.0001
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 16
  #--metric_for_best_model macro_f1
  --save_strategy epoch
  --evaluation_strategy epoch
  --logging_strategy epoch
  --seed 42
  #--weight_decay 0.1
  --save_total_limit 1
  --num_train_epochs 10
  --adapter_name vulgär
  --load_best_model_at_end True
  --class_weights True
  --pretrained_adapters_file ""
  --fusion_path ""
  )
python ./train_STadapter.py "${args[@]}" "$@"
