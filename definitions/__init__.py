#!/usr/bin/env python3

import csv
from dataclasses import dataclass
from collections import defaultdict
from nltk.corpus import wordnet
import numpy as np
from pathlib import Path
import torch

from util import EvalResult

@dataclass
class DefinitionExample:
    nonce_word: str
    orig_word: str
    definition: str
    good_context: str
    bad_context: str
    good_target: str
    bad_target: str

class MisraDefinitionTask:
    def __init__(self):
        random = np.random.RandomState(0)
        curr_path = Path(__file__).parent
        data_path = curr_path / Path("data_misra/val_1ns.csv")

        definitions = {}
        good_examples = defaultdict(list)
        bad_examples = defaultdict(list)

        with open(data_path) as reader:
            next(reader)
            for line in reader:
                _, label, concept, category, attr, _ = line.strip().split(",")
                if len(wordnet.synsets(concept)) == 0:
                    continue
                synset = wordnet.synsets(concept)[0]

                definitions[concept] = synset.definition()
                (good_examples if int(label) else bad_examples)[concept].append(attr)

        examples = []
        for word in sorted(definitions.keys()):
            nonce = "wug"
            defn = definitions[word]
            if len(good_examples[word]) == 0 or len(bad_examples[word]) == 0:
                continue
            good_ex = "A wug " + random.choice(good_examples[word]) + "."
            bad_ex = "A wug " + random.choice(bad_examples[word]) + "."
            print("A wug is " + defn)
            print(good_ex)
            print(bad_ex)
            print()
            example = DefinitionExample(
                nonce,
                word,
                "A wug is " + defn + ".",
                None,
                None,
                good_ex,
                bad_ex,
            )
            examples.append(example)


        self.train_examples = None
        random.shuffle(examples)
        self.eval_examples = examples[:20]

    def __iter__(self):
        for example in self.eval_examples:
            yield (
                (example.nonce_word, example.definition),
                (example.good_target, example.bad_target),
            )

    def evaluate(self, model, targets):
        good_tgt, bad_tgt = targets
        with torch.no_grad():
            diff = (
                model(bad_tgt) - model(good_tgt)
            )
        return EvalResult({"diff": diff, "acc": int(diff > 0)})

    def make_evaluation_prompt(self, supervision):
        word, definition = supervision
        return (
            definition 
            + f" Here is an example of a sentence using the word {word}:"
        )

    def make_extrapolation_prompt(self, supervision):
        word, definition = supervision
        return (
            definition
            + f" Here is an example of a sentence using the word {word}:"
        )


class WinodictDefinitionTask:
    def __init__(self, easy):
        random = np.random.RandomState(0)

        curr_path = Path(__file__).parent
        data_path = curr_path / Path("data_winodict/winodict/winograd_prob1_of_5.csv")

        examples = []
        with open(data_path) as reader:
            csv_reader = csv.reader(reader)
            next(csv_reader)
            for line in csv_reader:
                assert len(line) == 11
                orig = line[1]
                nonce = line[2]
                defn = line[6]
                ex = line[7]
                ans1 = line[8]
                ans2 = line[9]
                correct = line[10]
                if correct == "1":
                    ans1, ans2 = ans2, ans1
                if ". _" not in ex:
                    if ans1.startswith("The"):
                        ans1 = ans1.replace("The", "the")
                    if ans2.startswith("The"):
                        ans2 = ans2.replace("The", "the")

                prefix, suffix = ex.split("_")

                prefix_pop_good = prefix + ans1 #ex.replace("_", ans1)
                prefix_pop_bad = prefix + ans2 #ex.replace("_", ans2)

                #example = DefinitionExample(
                #    nonce,
                #    orig,
                #    defn 
                #        + " Here is an example of a sentence using the word " 
                #        + nonce 
                #        + ":", 
                #    "\n" + ex_pop_good,
                #    "\n" + ex_pop_bad,
                #)
                example = DefinitionExample(
                    nonce,
                    orig,
                    defn,
                    prefix_pop_good,
                    prefix_pop_bad,
                    suffix,
                )
                if nonce == "voin":
                    continue
                examples.append(example)

        #self.train_examples = examples[:10]
        self.train_examples = None
        #self.eval_examples = examples[10:110]
        self.eval_examples = examples[:100]

        if easy:
            assert False
            for example in self.eval_examples:
                easy_distractor = None
                for example2 in random.permutation(self.eval_examples):
                    if example2 == example:
                        continue
                    if example2.nonce_word != example.nonce_word:
                        continue
                    easy_distractor = example2
                    break
                assert easy_distractor is not None
                example.bad_tgt = easy_distractor.good_tgt

    def __iter__(self):
        for example in self.eval_examples:
            yield (
                (example.nonce_word, example.definition),
                (example.good_context, example.bad_context, example.target),
            )

    def evaluate(self, model, targets):
        good_ctx, bad_ctx, tgt = targets
        with torch.no_grad():
            diff = (
                (model(bad_ctx + tgt) - model(bad_ctx))
                - (model(good_ctx + tgt) - model(good_ctx))
            )
        return EvalResult({"diff": diff, "acc": int(diff > 0)})

    def make_evaluation_prompt(self, supervision):
        word, definition = supervision
        return (
            definition 
            + f" Here is an example of a sentence using the word {word}:"
        )

    def make_extrapolation_prompt(self, supervision):
        word, definition = supervision
        return (
            definition
            + f" Here is an example of a sentence using the word {word}:"
        )
