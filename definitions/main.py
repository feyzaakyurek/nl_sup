#!/usr/bin/env python3

import csv
from dataclasses import dataclass
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch import nn, optim

from gptcache import GPTCache

random = np.random.RandomState(0)

@dataclass
class Example:
    nonce_word: str
    orig_word: str
    context: str
    good_tgt: str
    bad_tgt: str

def make_fewshot(examples):
    out = []
    for ex1 in examples:
        for ex2 in random.permutation(examples):
            if ex1.nonce_word == ex2.nonce_word:
                continue
            new_ex = Example(
                ex1.nonce_word,
                ex1.orig_word,
                ex2.context + ex2.good_tgt + "\n\n" + ex1.context,
                ex1.good_tgt,
                ex1.bad_tgt
            )
            out.append(new_ex)
            break
    return out

simple_examples = []
defn_examples = []
selfsup_propose_examples = []
selfsup_use_examples = []

gpt = GPTCache("gpt_cache.json")

with open("data_winodict/winodict/winograd_prob1_of_5.csv") as reader:
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

        ex_pop_good = ex.replace("_", ans1)
        ex_pop_bad = ex.replace("_", ans2)
        simple_example = Example(nonce, orig, "", ex_pop_good, ex_pop_bad)
        defn_example = Example(nonce, orig, defn, " " + ex_pop_good, " " + ex_pop_bad)
        simple_examples.append(simple_example)
        defn_examples.append(defn_example)

        selfsup_propose_example = Example(
            nonce,
            orig,
            defn 
                + " Here is an example of a sentence using the word " 
                + nonce 
                + ":", 
            "\n" + ex_pop_good,
            "\n" + ex_pop_bad,
        )
        selfsup_propose_examples.append(selfsup_propose_example)



selfsup_propose_examples_easy = []
for example in selfsup_propose_examples:
    example2 = None
    for example_ in np.random.permutation(selfsup_propose_examples):
        if example_ == example:
            continue
        if example_.nonce_word != example.nonce_word:
            continue
        if example_.good_tgt == example.bad_tgt:
            continue
        example2 = example_
        break
    assert example2 is not None
    new_example = Example(
        example.nonce_word,
        example.orig_word,
        example.context,
        example.good_tgt,
        example2.good_tgt
    )
    selfsup_propose_examples_easy.append(new_example)


#for example in make_fewshot(selfsup_propose_examples):
#    usage = gpt.generate(example.context, max_tokens=40)
#    if "." in usage:
#        usage = usage[:usage.index(".")+1]
#    usage = usage.strip()
#    new_context = (
#        example.context.split("\n\n")[1].replace(
#            "is an example of a sentence",
#            "are some examples of sentences"
#        ) 
#        + "\n" 
#        + usage
#    )
#    selfsup_use_example = Example(
#        example.nonce_word, new_context, example.good_tgt, example.bad_tgt
#    )
#    selfsup_use_examples.append(selfsup_use_example)
#
#acc = {}
#for name, dataset in [
#        ("simple", simple_examples),
#        ("defn", defn_examples),
#        ("selfsup", selfsup_use_examples),
#]:
#    acc[name] = 0
#    for example in make_fewshot(dataset):
#        good_score = gpt.score(example.context, example.good_tgt)
#        bad_score = gpt.score(example.context, example.bad_tgt)
#        acc[name] += int(good_score > bad_score)
#    acc[name] /= len(dataset)
#
#print(acc)

#for example in make_fewshot(selfsup_propose_examples_easy)[0:20]:
for example in selfsup_propose_examples_easy:
    ft_examples = []
    i = 0
    while len(ft_examples) < 100:
        usage = gpt.generate(example.context, max_tokens=40, index=i)
        i += 1
        if "." in usage:
            usage = usage[:usage.index(".")+1]
        usage = usage.strip()
        if usage in ft_examples:
            continue
        ft_examples.append(usage)
        #print(usage)

    print()
    print(example.good_tgt.strip())
    print(example.bad_tgt.strip())



    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token
    #orig_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    #orig_model.eval()
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").cuda()
    model.eval()


    orig_bad = example.bad_tgt.replace(example.nonce_word, example.orig_word)
    orig_good = example.good_tgt.replace(example.nonce_word, example.orig_word)
    orig_bad_batch = tokenizer(orig_bad, return_tensors="pt").input_ids.cuda()
    orig_good_batch = tokenizer(orig_good, return_tensors="pt").input_ids.cuda()
    nll_orig_good = model(orig_good_batch, labels=orig_good_batch).loss
    nll_orig_bad = model(orig_bad_batch, labels=orig_bad_batch).loss
    #if nll_orig_bad < nll_orig_good:
    #    print("original model failed on [", orig_bad, "] < [", orig_good, "]")
    #    continue


    opt = optim.AdamW(model.parameters(), lr=1e-5)
    #opt = optim.SGD(model.parameters(), lr=1e-4)
    kl = nn.KLDivLoss()
    for i in range(50):
        ft_batch = tokenizer(np.random.choice(ft_examples, size=4).tolist(), return_tensors="pt", padding=True).input_ids.cuda()
        bad_batch = tokenizer(example.bad_tgt.strip(), return_tensors="pt").input_ids.cuda()
        good_batch = tokenizer(example.good_tgt.strip(), return_tensors="pt").input_ids.cuda()
        nll_good = model(good_batch, labels=good_batch).loss
        nll_bad = model(bad_batch, labels=bad_batch).loss
        if i == 0 or i == 49:
            print(nll_bad.item() - nll_good.item())
        model_preds = model(ft_batch, labels=ft_batch)
        pred_loss = model_preds.loss
        #with torch.no_grad():
        #    orig_logits = orig_model(ft_batch, labels=ft_batch).logits.softmax(dim=2)
        #keep_loss = kl(
        #    model_preds.logits.log_softmax(dim=2),
        #    orig_logits,
        #)
        #loss = 0.1 * pred_loss + keep_loss
        loss = pred_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    print()

    del model
    del opt
    torch.cuda.empty_cache()
