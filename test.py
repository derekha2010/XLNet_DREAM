#!pip install transformers
#!pip install sentencepiece
from transformers import XLNetTokenizer, XLNetForMultipleChoice
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.utils import DataProcessor, InputFeatures
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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
output_model_file = 'model/pytorch_model_3epoch_72_len256.bin'
train_batch_size = train_batch_size // gradient_accumulation_steps

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        
def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # Note: XLNet: A + [SEP] + B + [SEP] + C + [SEP] [CLS]
    # CLS token id for XLNet of 2 
    # pad on left for XLNet 
    cls_tok_id = 2
    pad_tok_id = 4 
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    print("#examples", len(examples))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None

        tokens_c = None
        
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if example.text_c:
            tokens_c = tokenizer.tokenize(example.text_c)

        if tokens_c:
            _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
            tokens_b = tokens_c + [sep_token] + tokens_b
        elif tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(sep_token)
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append(sep_token)
            segment_ids.append(1)

        tokens.append(cls_token)
        segment_ids.append(cls_tok_id)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length. 
        # pad on left !!! 
        # pad_token = 0
        # pad_segment_id = 4
        padding_length = max_seq_length - len(input_ids)
        input_ids = ([0]*padding_length) + input_ids
        input_mask = ([0]*padding_length) + input_mask
        segment_ids = ([pad_tok_id]*padding_length) + segment_ids

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        token_type_ids=segment_ids,
                        label=label_id))
        ## three egs per list 
        if len(features[-1]) == n_class:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()
            
class dreamProcessor(DataProcessor):
    def __init__(self):
        random.seed(random_seed)
        self.D = [[], [], []]

        for sid in range(3):
            ## Note: assuming data folder stored in the same directory 
            with open(["data/train.json", "data/dev.json", "data/test.json"][sid], "r") as f:
                data = json.load(f)
                if sid == 0:
                    random.shuffle(data)
                for i in range(len(data)):
                    for j in range(len(data[i][1])):
                        # shouldn't do lower case, since we are using cased model                         
                        # d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                        d = ['\n'.join(data[i][0]), data[i][1][j]["question"]]
                        for k in range(len(data[i][1][j]["choice"])):
                            # d += [data[i][1][j]["choice"][k].lower()]
                            d += [data[i][1][j]["choice"][k]]
                        # d += [data[i][1][j]["answer"].lower()] 
                        d += [data[i][1][j]["answer"]] 
                        self.D[sid] += [d]
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            for k in range(3):
                # each d: passage, question, choice * 3, answer 
                if data[i][2+k] == data[i][5]:
                    answer = str(k)
                    
            label = convert_to_unicode(answer)

            for k in range(3):
                guid = "%s-%s-%s" % (set_type, i, k)
                ## passage 
                text_a = convert_to_unicode(data[i][0])
                ## choice 
                text_b = convert_to_unicode(data[i][k+2])
                ## question 
                text_c = convert_to_unicode(data[i][1])
                examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))
            
        return examples

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('gpu count:', n_gpu)
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(random_seed)

    model_state_dict = torch.load(output_model_file, map_location=device)
    model = XLNetForMultipleChoice.from_pretrained('xlnet-base-cased', state_dict=model_state_dict)
    logger.info("Trained model: {} loaded.".format(output_model_file))

    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    ## note: no weight decay according to XLNet paper 
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    processor = dreamProcessor()
    label_list = processor.get_labels()

    eval_examples = processor.get_test_examples('')
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []
    
    for f in eval_features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(n_class):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].attention_mask)
            segment_ids[-1].append(f[i].token_type_ids)
        label_id.append([f[0].label])                

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.long)

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
        
if __name__ == "__main__":
    main()