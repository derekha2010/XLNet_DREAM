#!pip install transformers
#!pip install sentencepiece
from transformers import XLNetTokenizer, XLNetForMultipleChoice
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils_multiple_choice import (convert_examples_to_features, processors)

import argparse
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

#train_batch_size = 1
#gradient_accumulation_steps = 1
#num_train_epochs = 5
#num_warmup_steps = 0
#max_seq_length = 256
eval_batch_size = 1
#learning_rate = 1e-5
random_seed = 42

#data_path='datasets/MCTest'
#output_dir='model-mc500'
dataset_map={'mc500':'datasets/MCTest', 'mc160':'datasets/MCTest','dream':'datasets/DREAM','race':'datasets/RACE'}

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

def load_and_cache_examples(data_dir, task, model, tokenizer, evaluate=False, test=False):
    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = 'dev'
    elif test:
        cached_mode = 'test'
    else:
        cached_mode = 'train'
    assert (evaluate == True and test == True) == False
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        cached_mode,
        model,
        str(max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(data_dir)
        elif test:
            examples = processor.get_test_examples(data_dir)
        else:
            examples = processor.get_train_examples(data_dir)
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            max_seq_length,
            tokenizer,
            pad_on_left=True,  # pad on the left for xlnet
            pad_token_segment_id=4
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
    
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The model file")                    
    parser.add_argument("--seq_length",
                        default=None,
                        type=int,
                        required=True,
                        help="seq_length")                             
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        required=True,
                        help="model")
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="dataset")
    args = parser.parse_args()
    global max_seq_length
    max_seq_length = args.seq_length
    output_model_file = args.model_file

    output_dir='model-'+args.dataset
    data_path=dataset_map[args.dataset]
    
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
    model = XLNetForMultipleChoice.from_pretrained(args.model, state_dict=model_state_dict)
    model.to(device)
    tokenizer = XLNetTokenizer.from_pretrained(args.model)
    
    eval_data = load_and_cache_examples(data_path, args.dataset, args.model, tokenizer, test=True)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    
    logger.info("***** Running Evaluation *****")
    logger.info("  Num examples = %d", len(eval_data))
    logger.info("  Batch size = %d", eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            eval_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        tmp_eval_loss = eval_output.loss
        logits = eval_output.logits
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        
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

    output_eval_file = os.path.join(output_dir, "{}_{}_{}_results.txt".format(args.dataset, args.model, max_seq_length))
    with open(output_eval_file, "a+") as writer:
        writer.write("Test:\n")
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()