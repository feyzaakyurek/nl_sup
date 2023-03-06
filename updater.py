import numpy as np
from torch import optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
tokenizer.pad_token = tokenizer.eos_token

def _make_model():
    return GPT2LMHeadModel.from_pretrained("gpt2-xl").cuda()

class GPT2ModelWrapper:
    def __init__(self, model, preprocessor=lambda x: x):
        self.model = model
        self.preprocessor = preprocessor

    def __call__(self, seq):
        seq_preproc = self.preprocessor(seq)
        seq_tok = tokenizer(seq_preproc, return_tensors="pt").input_ids.cuda()
        return self.model(seq_tok, labels=seq_tok).loss.item()

class NullUpdater:
    def __init__(self):
        self.model = _make_model()
        self.model.eval()

    def __call__(self, task, supervision, targets):
        return GPT2ModelWrapper(self.model)

class PromptUpdater:
    def __init__(self):
        self.model = _make_model()
        self.model.eval()

    def __call__(self, task, supervision, targets):
        def preprocessor(seq):
            return task.make_evaluation_prompt(supervision) + " " + seq
        return GPT2ModelWrapper(self.model, preprocessor)

class ExampleUpdater:
    def __init__(self, extrapolator):
        self.extrapolator = extrapolator
        self.random = np.random.RandomState(0)

    def __call__(self, task, supervision, targets):
        extrap_examples = self.extrapolator(task, supervision)
        print("\n".join(extrap_examples))
        model = _make_model()
        model.eval()
        opt = optim.AdamW(model.parameters(), lr=3e-6)
        for i in range(0, len(extrap_examples), 4):
            ft_batch = tokenizer(
                #self.random.choice(extrap_examples, size=4).tolist(),
                extrap_examples[i:i+4],
                return_tensors="pt",
                padding=True,
            ).input_ids.cuda()
            print(task.evaluate(GPT2ModelWrapper(model), targets))
            loss = model(ft_batch, labels=ft_batch).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        del opt
        return GPT2ModelWrapper(model)
