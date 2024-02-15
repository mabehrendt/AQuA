from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterTrainer
from data import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from inference_parallel import task2identifier
import argparse


def predict(dataloader, model, out_put_path, dataset, task):
    # create a dictionary with a list for each label
    label_num = 4
    output_dic = []
#    for i in range(label_num):
#        output_dic[i] = []
    for id, batch in enumerate(tqdm(dataloader)):
        # print(batch)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predictions = outputs.logits
        #print(predictions.shape)
        #if label_num == 1:
        #    probs = np.squeeze(predictions, axis=1).tolist()
        #    output_dic[0].extend(probs)
        #elif label_num == 2:
        #    probs = torch.sigmoid(torch.tensor(predictions)).tolist()
        #else:
        #probs = F.softmax(torch.tensor(predictions), dim=-1).tolist()
        prediction = torch.argmax(predictions, axis=1).detach().numpy()
        output_dic.extend(prediction)
        #if label_num > 1:
        #for i in range(label_num):
        #    output_dic[i].extend([el[i] for el in probs])
   # if label_num == 1:
   #     dataset[task] = output_dic[0]
   # else:
    dataset[task] = output_dic

    dataset.to_csv(out_put_path, sep="\t", index=False)


if __name__ == '__main__':
    # read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('testdata', type=str,
                        help='path to the test data')
    parser.add_argument('text_col', type=str, help="column name of text column")
    parser.add_argument('batch_size', type=int)
    parser.add_argument('task', type=str, help="task name or name of quality dimension")
    parser.add_argument("output_path", type=str, help="path to output file")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoAdapterModel.from_pretrained("roberta-base")
    adapter_name = model.load_adapter(task2identifier[args.task], source="hf", set_active=True)

    test = InferenceDataset(path_to_dataset=args.testdata, tokenizer=tokenizer, text_col=args.text_col)
    dataloader = DataLoader(test, batch_size=args.batch_size)
    predict(dataloader=dataloader, model=model, out_put_path=args.output_path,
            dataset=test.dataset, task=args.task)
