import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import argparse

if __name__ == '__main__':
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str,
                        help='path to the csv file to be translated')
    parser.add_argument('dataset_name', type=str,
                        help='name of the dataset')
    parser.add_argument('translation_col', type=str, help='column name of text column to be translated')
    parser.add_argument('output_path', type=str, help='path to output file')
    parser.add_argument('--sep', type=str, default='\t', required=False, help='separaptor used in the csv file')
    args = parser.parse_args()

    # load models
    model_name = 'facebook/wmt19-en-de'
    tokenizer = FSMTTokenizer.from_pretrained(model_name)
    model = FSMTForConditionalGeneration.from_pretrained(model_name)

    # load data
    df = pd.read_csv(args.dataset_path,sep=args.sep)

    # translate
    df['comment_de'] = ''
    for i in range(len(df)):
        input_ids = tokenizer.encode(df[args.translation_col][i], return_tensors='pt')
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        df['comment_de'][i] = decoded

    # save to csv
    # use \t as seperator to match our inference scripts
    df.to_csv(args.output_path+args.dataset_name+'_de_translated.csv',sep='\t',index=False)
