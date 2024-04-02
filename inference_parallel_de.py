from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterTrainer
from data import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import argparse
import os
from utils import get_dynamic_parallel

# check for GPUs or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU in use:')
else:
    print('using the CPU')
    device = torch.device("cpu")

def predict(dataloader, model, dataset, output_path, task2identifier):
    """
    Predict AQuA scores for a dataset.

    Arguments:
    dataloader  – dataloader for inference
    model   – model used for inference
    dataset    – dataset for inference
    output_path    – path to the output csv file
    task2identifier    – dictionary including adapters and their path 
    """
    output_dic = {}
    for k, v in task2identifier.items():
        output_dic[k] = []
    for id, batch in enumerate(tqdm(dataloader)):
        # output length = num adapters
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        for i in range(len(outputs)):
            task = list(task2identifier.keys())[i]
            # gives predictions for one batch for one adapter
            predictions = outputs[i].logits
            prediction = torch.argmax(predictions, axis=1).detach().numpy()
            output_dic[task].extend(prediction)
    for task, preds in output_dic.items():
        dataset[task] = preds
    score = dataset[list(task2identifier.keys())].dot(weights)
    # normalize the score
    dataset["score"] = normalize_scores(score)
    dataset.to_csv(output_path, sep="\t", index=False)
    
def normalize_scores(scores, bound=5):
    """
    Returns the normalized scores in an interval between 0 and bound.

    Arguments:
    scores  – scores to normalize
    bound   – upper interval bound
    """
    return ((scores-minval)/minmaxdif)*bound
    
weights = [0.20908452, 0.18285757, -0.11069402, 0.29000763, 0.39535126,
        0.14655912, -0.07331445, -0.03768367, 0.07019062, -0.02847408,
        0.21126469, -0.02674237, 0.01482095, 0.00732909, -0.01900971,
        -0.04995486, -0.05884586, -0.15170863, 0.02934227, 0.10628146]

maxval = 4.989267539999999
minval = -1.66928295
minmaxdif = maxval-minval

task2identifier = {"relevance": "trained adapters/relevance",
                   "fact": "trained adapters/fact",
                   "opinion": "trained adapters/opinion",
                   "justification": "trained adapters/justification",
                   "solproposal": "trained adapters/solproposal",
                   "addknowledge": "trained adapters/addknowledge"},
                   "question": "trained adapters/question",

                   "refusers": "trained adapters/refusers",
                   "refmedium": "trained adapters/refmedium",
                   "refcontents": "trained adapters/refcontents",
                   "refpersonal": "trained adapters/refpersonal",
                   "refformat": "trained adapters/refformat",

                   "address": "trained adapters/address",
                   "respect": "trained adapters/respect",
                   "screaming": "trained adapters/screaming",
                   "vulgar": "trained adapters/vulgar",
                   "insult": "trained adapters/insult",
                   "sarcasm": "trained adapters/sarcasm",
                   "discrimination": "trained adapters/discrimination",

                   "storytelling": "trained adapters/storytelling"}


task2weight = {"relevance": 0.20908452,
                   "fact":0.18285757,
                   "opinion": -0.11069402,
                   "justification": 0.29000763,
                   "solproposal":0.39535126,
                   "addknowledge": 0.14655912,
                   "question":-0.07331445,

                   "refusers":-0.03768367,
                   "refmedium":0.07019062,
                   "refcontents":-0.02847408,
                   "refpersonal":0.21126469,
                   "refformat": -0.02674237,

                   "address": 0.01482095,
                   "respect": 0.00732909,
                   "screaming": -0.01900971,
                   "vulgar": -0.04995486,
                   "insult": -0.05884586,
                   "sarcasm":-0.15170863,
                   "discrimination":0.02934227,

                   "storytelling": 0.10628146}

if __name__ == '__main__':
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('inference_data', type=str,
                        help='path to the test data')
    parser.add_argument('text_col', type=str, help="column name of text column")
    parser.add_argument('batch_size', type=int)
    parser.add_argument("output_path", type=str, help="path to output file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoAdapterModel.from_pretrained("bert-base-multilingual-cased").to(device)
    adapter_counter = 0

    for k, v in task2identifier.items():
        print("loading adapter %s as adapter %d" % (k, adapter_counter))
        model.load_adapter(v, load_as="adapter%d" % adapter_counter, with_head=True,
                           set_active=True, source="hf")
        adapter_counter += 1
    print("loaded %d adapters" % adapter_counter)
    adapter_setup = get_dynamic_parallel(adapter_number=adapter_counter)
    model.active_adapters = adapter_setup
    model.eval()
    test = InferenceDataset(path_to_dataset=args.inference_data, tokenizer=tokenizer, text_col=args.text_col)
    dataloader = DataLoader(test, batch_size=args.batch_size)
    predict(dataloader=dataloader, model=model, dataset=test.dataset,
                output_path=args.output_path, task2identifier=task2identifier)
