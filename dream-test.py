#!pip install sentencepiece
#!pip install transformers

from transformers import XLNetTokenizer, XLNetForMultipleChoice
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils_multiple_choice import (convert_examples_to_features, processors)

import os
import torch
import random
import json
import logging
import numpy as np

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

n_class = 3
train_batch_size = 32
gradient_accumulation_steps = 32
num_train_epochs = 4
num_warmup_steps = 120
max_seq_length = 256
eval_batch_size = 1
learning_rate = 1e-5
random_seed = 42

output_dir='model-dream'
output_model_file = 'trained/dream_xlnet-large-cased_len256_72.bin'

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('gpu count:', n_gpu)
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    model_state_dict = torch.load(output_model_file, map_location=device)
    model = XLNetForMultipleChoice.from_pretrained('xlnet-large-cased', state_dict=model_state_dict)
    logger.info("Trained model: {} loaded.".format(output_model_file))

    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    ## note: no weight decay according to XLNet paper 
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    processor = processors['dream']()
    label_list = processor.get_labels()

    eval_examples = processor.get_test_examples('')
    eval_features = convert_examples_to_features(
            eval_examples,
            label_list,
            max_seq_length,
            tokenizer,
            pad_on_left=True,  # pad on the left for xlnet
            pad_token_segment_id=4
        )

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)

    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            eval_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, n_class=n_class)
        tmp_eval_loss = eval_output.loss
        logits = eval_output.logits
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        for i in range(len(logits)):
            logits_all += [logits[i]]
        
        tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        output_eval_file = os.path.join(output_dir, "results.txt")
    with open(output_eval_file, "a+") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))
if __name__ == "__main__":
    main()