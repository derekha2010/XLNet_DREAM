# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function

import random
import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
from transformers import PreTrainedTokenizer

random_seed = 42

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, 'train/high')
        middle = os.path.join(data_dir, 'train/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, 'dev/high')
        middle = os.path.join(data_dir, 'dev/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, 'test/high')
        middle = os.path.join(data_dir, 'test/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw['answers'][i]) - ord('A'))
                question = data_raw['questions'][i]
                options = data_raw['options'][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article], # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth))
        return examples

class DreamProcessor(DataProcessor):
    def get_file(self, file_path, ramdom=False):
        D = []
        with open(file_path, "r") as f:
            data = json.load(f)
            if ramdom:
                random.seed(random_seed)
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
                    D += [d]
        return D
        
    def get_train_examples(self, data_dir):
        return self._create_examples(
                get_file(os.path.join(data_dir, 'train.json'), random=True), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                get_file(os.path.join(data_dir, 'test.json')), "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                get_file(os.path.join(data_dir, 'dev.json')), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            label = None
            for k in range(3):
                if data[i][2+k] == data[i][5]:
                    label = str(k)
            examples.append(
                    InputExample(example_id=guid, question=data[i][1], contexts=[data[i][0],data[i][0],data[i][0]], endings=[data[i][2],data[i][3],data[i][4]], label=label))
        return examples

class MC160Processor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "MCTest/mc160.train.tsv")), self._read_csv(os.path.join(data_dir, "MCTest/mc160.train.ans")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "MCTest/mc160.dev.tsv")), self._read_csv(os.path.join(data_dir, "MCTest/mc160.dev.ans")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "MCTest/mc160.test.tsv")), self._read_csv(os.path.join(data_dir, "MCTestAnswers/mc160.test.ans")), "test")
        
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


    def _create_examples(self, lines: List[List[str]], ans_lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            line = lines[i]
            ans_line = ans_lines[i]
            for j in range(4):
                guid = "%s-%s" % (line[0], j)
                examples.append(
                    InputExample(
                        example_id=guid,
                        question=line[3+j*5],  # in the swag dataset, the
                        # common beginning of each
                        # choice is stored in "sent2".
                        contexts = [line[2], line[2], line[2], line[2]],
                        endings = [line[3+j*5+1], line[3+j*5+2], line[3+j*5+3], line[3+j*5+4]],
                        label= str(ord(ans_line[j]) - ord("A"))
                    )
                )

        return examples

class MC500Processor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "MCTest/mc500.train.tsv")), self._read_csv(os.path.join(data_dir, "MCTest/mc500.train.ans")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "MCTest/mc500.dev.tsv")), self._read_csv(os.path.join(data_dir, "MCTest/mc500.dev.ans")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "MCTest/mc500.test.tsv")), self._read_csv(os.path.join(data_dir, "MCTestAnswers/mc500.test.ans")), "test")
        
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


    def _create_examples(self, lines: List[List[str]], ans_lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            line = lines[i]
            ans_line = ans_lines[i]
            for j in range(4):
                guid = "%s-%s" % (line[0], j)
                examples.append(
                    InputExample(
                        example_id=guid,
                        question=line[3+j*5],  # in the swag dataset, the
                        # common beginning of each
                        # choice is stored in "sent2".
                        contexts = [line[2], line[2], line[2], line[2]],
                        endings = [line[3+j*5+1], line[3+j*5+2], line[3+j*5+3], line[3+j*5+4]],
                        label= str(ord(ans_line[j]) - ord("A"))
                    )
                )

        return examples
        
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
            )
            if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
                logger.info('Attention! you are cropping tokens (swag task is ok). '
                        'If you are training ARC and RACE and you are poping question + options,'
                        'you need to try to use a bigger max seq length!')

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))


        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            )
        )

    return features

processors = {
    "mc160": MC160Processor,
    "mc500": MC500Processor,
    "dream": DreamProcessor,
    "race": RaceProcessor
}

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "mc160", 4,
    "mc500", 4,
    "dream", 3,
    "race", 4
}