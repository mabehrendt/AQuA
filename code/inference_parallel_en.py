from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterTrainer
from data import InferenceDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import argparse
import os
from utils import get_dynamic_parallel

# choose GPU 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# check for GPUs or CPU
#if torch.cuda.is_available():
#    device = torch.device("cuda")
#    print('GPU in use:')
#else:
#    print('using the CPU')
device = torch.device("cpu")

def predict(dataloader, model, dataset, output_path, task2identifier):
    output_dic = {}
    for k, v in task2identifier.items():
        output_dic[k] = []
    for id, batch in enumerate(tqdm(dataloader)):
        for k, v in batch.items():
            #if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        # output length = num adapters
        outputs = model(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda())
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
    dataset["score"] = ((score-minval)/minmaxdif)*5
    dataset.to_csv(output_path, sep="\t", index=False)
    
weights = torch.tensor([0.01482095,  0.29000763, -0.05884586, -0.02674237,
       -0.02847408,  0.07019062, -0.03768367, 0.21126469, 0.02934227,
       -0.07331445,  0.39535126, -0.11069402,  0.00732909,
       -0.15170863, -0.01900971, 0.10628146,  0.18285757,  0.20908452, -0.04995486,
        0.14655912], device=device)

maxval = torch.tensor([4.989267539999999], device=device)
minval = torch.tensor([-1.66928295], device=device)
minmaxdif = maxval-minval

task2identifier = {"anrede": "trained adapters/trained_adapters_en/anrede",
                   "begründung": "trained adapters/trained_adapters_en/begründung",
                   "beleidigung": "trained adapters/trained_adapters_en/beleidigung",
                   "bezugform": "trained adapters/trained_adapters_en/bezugform",
                   "bezuginhalt": "trained adapters/trained_adapters_en/bezuginhalt",
                   "bezugmedium": "trained adapters/trained_adapters_en/bezugmedium",
                   "bezugnutzer": "trained adapters/trained_adapters_en/bezugnutzer",
                   "bezugpersönlich": "trained adapters/trained_adapters_en/bezugpersönlich",
                   "diskriminierung": "trained adapters/trained_adapters_en/diskriminierung",
                   "frage": "trained adapters/trained_adapters_en/frage",
                   "lösungsvorschlag": "trained adapters/trained_adapters_en/lösungsvorschlag",
                   "meinung": "trained adapters/trained_adapters_en/meinung",
                   "respekt": "trained adapters/trained_adapters_en/respekt",
                   "sarkasmus": "trained adapters/trained_adapters_en/sarkasmus",
                   "schreien": "trained adapters/trained_adapters_en/schreien",
                   "storytelling": "trained adapters/trained_adapters_en/storytelling",
                   "tatsache": "trained adapters/trained_adapters_en/tatsache",
                   "themenbezug": "trained adapters/trained_adapters_en/themenbezug",
                   "vulgär": "trained adapters/trained_adapters_en/vulgär",
                   "zusatzwissen": "trained adapters/trained_adapters_en/zusatzwissen"}

task2weight = {"anrede": 0.01482095,
                   "begründung": 0.29000763,
                   "beleidigung": -0.05884586,
                   "bezugform": -0.02674237,
                   "bezuginhalt":-0.02847408,
                   "bezugmedium":0.07019062,
                   "bezugnutzer":-0.03768367,
                   "bezugpersönlich":0.21126469,
                   "diskriminierung":0.02934227,
                   "frage":-0.07331445,
                   "lösungsvorschlag":0.39535126,
                   "meinung": -0.11069402,
                   "respekt": 0.00732909,
                   "sarkasmus":-0.15170863,
                   "schreien": -0.01900971,
                   "storytelling": 0.10628146,
                   "tatsache":0.18285757,
                   "themenbezug": 0.20908452,
                   "vulgär": -0.04995486,
                   "zusatzwissen": 0.14655912}

if __name__ == '__main__':
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('testdata', type=str,
                        help='path to the test data')
   # parser.add_argument('separator', type=str,
   #                     help='separator in the csv file')
    parser.add_argument('text_col', type=str, help="column name of text column")
    parser.add_argument('batch_size', type=int)
    parser.add_argument("output_path", type=str, help="path to output file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoAdapterModel.from_pretrained("roberta-base").to(device)
    model.eval()
    adapter_counter = 0

    for k, v in task2identifier.items():
        print("loading adapter %s as adapter %d" % (k, adapter_counter))
        model.load_adapter(v, load_as="adapter%d" % adapter_counter, with_head=True,
                           set_active=True, source="hf")
        adapter_counter += 1

            #model.set_active_adapters([i for i in range(adapter_counter)])
    print("loaded %d adapters" % adapter_counter)
    adapter_setup = get_dynamic_parallel(adapter_number=adapter_counter)
    model.active_adapters = adapter_setup
    test = InferenceDataset(path_to_dataset=args.testdata, tokenizer=tokenizer, text_col=args.text_col)
    dataloader = DataLoader(test, batch_size=args.batch_size)
                           #collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    predict(dataloader=dataloader, model=model, dataset=test.dataset,
                output_path=args.output_path, task2identifier=task2identifier)
