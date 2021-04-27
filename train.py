#!pip install sentencepiece
#!pip install transformers
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

#gradient_accumulation_steps =
#num_train_epochs =
#num_warmup_steps =
#max_seq_length =
train_batch_size = 1
eval_batch_size = 1
learning_rate = 1e-5
random_seed = 42

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
                        required=False,
                        help="The model file")
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="Batch size")                        
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
    parser.add_argument("--epochs",
                        default=5,
                        type=int,
                        required=False,
                        help="Epochs")     
                        
    args = parser.parse_args()
    global max_seq_length
    max_seq_length = args.seq_length

    global num_train_epochs
    num_train_epochs = args.epochs
    
    output_dir='model-'+args.dataset
    data_path=dataset_map[args.dataset]
    gradient_accumulation_steps = args.batch_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info('gpu count: %d', n_gpu)
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(random_seed)

    os.makedirs(output_dir, exist_ok=True)
    if(args.model_file):
        model_state_dict = torch.load(args.model_file, map_location=device)
        model = XLNetForMultipleChoice.from_pretrained(args.model, state_dict=model_state_dict, dropout=0, summary_last_dropout=0)
    else:
        ## note: dropout rate set to zero, increased accuracy
        model = XLNetForMultipleChoice.from_pretrained(args.model, dropout=0, summary_last_dropout=0)
    model.to(device)
    tokenizer = XLNetTokenizer.from_pretrained(args.model)
    
    no_decay = ['bias', 'LayerNorm.weight']
    ## note: no weight decay according to XLNet paper 
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    ## note: Adam epsilon used 1e-6
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=learning_rate,
                        eps=1e-6)

    train_data = load_and_cache_examples(data_path, args.dataset, args.model, tokenizer)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    
    ## note: Used gradient accumulation steps to simulate larger batch size
    num_train_steps = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    
    ## note: Warmup proportion of 0.1
    num_warmup_steps = num_train_steps//10
    logger.info("  Num warmup steps = %d", num_warmup_steps)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Batch size = %d", train_batch_size*gradient_accumulation_steps)
    logger.info("  Num steps = %d", num_train_steps)
    
    global_step = 0

    for ep in range(int(num_train_epochs)):
        model.train()
        max_score = 0 
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            loss = output.loss
            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if step%800 == 0:
                logger.info("Training loss: {}, global step: {}".format(tr_loss/nb_tr_steps, global_step))
                
        eval_data = load_and_cache_examples(data_path, args.dataset, args.model, tokenizer, evaluate=True)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
        
        logger.info("***** Running Dev Evaluation *****")
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
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss/nb_tr_steps}
        logger.info(" Epoch: %d", (ep+1))
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        output_eval_file = os.path.join(output_dir, "{}_{}_{}_results.txt".format(args.dataset, args.model, max_seq_length))
        with open(output_eval_file, "a+") as writer:
            writer.write("Epoch: "+str(ep+1)+"\n")
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("\n")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, "{}_{}_{}_epoch{}_{}.bin".format(args.dataset, args.model, max_seq_length, ep+1, int(eval_accuracy*100)))
        torch.save(model_to_save.state_dict(), output_model_file)

    # testdata
    test_data = load_and_cache_examples(data_path, args.dataset, args.model, tokenizer, test=True)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)
    
    logger.info("***** Running Test Evaluation *****")
    logger.info("  Num examples = %d", len(test_data))
    logger.info("  Batch size = %d", eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
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