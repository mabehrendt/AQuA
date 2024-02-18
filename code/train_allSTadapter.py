import os

# choose GPU 
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# list of all adapters and column names to train
adapters = ['anrede','begründung','beleidigung','bezugform','bezuginhalt','bezugmedium',
            'bezugnutzer','bezugpersönlich','diskriminierung','frage',
            'lösungsvorschlag', 'meinung','respekt','sarkasmus',
            'schreien','storytelling','tatsache','themenbezug',
            'vulgär','zusatzwissen']
columns = ['c_anrede','c_begruend','c_beleid','c_bezform','c_bezinhalt','c_bezmedium',
            'c_beznutzer','c_bezpers','c_diskrim','c_frage',
            'c_loes','c_meinung','c_respekt','c_sarkzy',
            'c_schrei','c_story','c_tatsache','c_thembez',
            'c_vulg','c_zusw']

outputdir = "trained_adapters_multilingual_cased_en_40epochs/"

if __name__ == '__main__':
    for i in range(len(adapters)):
        # data arguments
        args = ("--data_dir data/kodie --label "+columns[i]+
              " --labels_num 4 --max_seq_length 256 --text_col c_text"
              " --output_dir ./trained\ adapters/"+outputdir+adapters[i]+
              # model arguments
              " --model_name_or_path bert-base-multilingual-cased"
              # training arguments
              " --learning_rate 0.0001"
              " --per_device_train_batch_size 16"
              " --per_device_eval_batch_size 16"
              #--metric_for_best_model macro_f1
              " --save_strategy epoch"
              " --evaluation_strategy epoch"
              " --logging_strategy epoch"
              " --seed 42"
              #--weight_decay 0.1
              " --save_total_limit 1"
              " --num_train_epochs 40"
              " --adapter_name "+adapters[i]+
              " --load_best_model_at_end True"
              " --class_weights True"
              " --pretrained_adapters_file None"
              " --fusion_path None")
        print(args)

        os.system("python train_STadapter.py " + args)
        