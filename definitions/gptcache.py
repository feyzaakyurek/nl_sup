import json
import openai
import os

# Read from open_key.txt
with open("open_key.txt") as reader:
    openai.api_key = reader.read().strip()

class GPTCache:
    def __init__(self, loc):
        self.cache_loc = loc
        self.engine = "text-davinci-002"
        if os.path.exists(loc):
            with open(loc) as reader:
                self.cache = json.load(reader)
        else:
            self.cache = {"scores": {}, "generations": {}}

    def query(self, utt):
        if utt in self.cache["scores"]:
            return self.cache["scores"][utt]
        print("calling API with", "[" + utt + "]")
        result = openai.Completion.create(
            engine=self.engine,
            prompt=utt,
            max_tokens=0,
            logprobs=0,
            echo=True,
        )
        self.cache["scores"][utt] = result
        with open(self.cache_loc, "w") as writer:
            json.dump(self.cache, writer)
        return result

    def score(self, context, pred):
        result = self.query(context + pred)
        assert len(result["choices"]) == 1
        result = result["choices"][0]
        offset = result["logprobs"]["text_offset"].index(len(context))
        tokens = result["logprobs"]["tokens"][offset:]
        assert "".join(tokens) == pred

        logprobs = result["logprobs"]["token_logprobs"][offset:]
        if logprobs[0] is None:
            logprobs = logprobs[1:]
        return sum(logprobs)

    def generate(self, context, max_tokens, index=0):
        if context in self.cache["generations"] and len(self.cache["generations"][context]) > index:
            return self.cache["generations"][context][index]
        print("calling API with", "[" + context + "]")
        result = openai.Completion.create(
            engine=self.engine,
            prompt=context,
            max_tokens=max_tokens,
        )
        generation = result["choices"][0]["text"]
        if context not in self.cache["generations"]:
            self.cache["generations"][context] = []
        self.cache["generations"][context].append(generation)
        with open(self.cache_loc, "w") as writer:
            json.dump(self.cache, writer)
        return generation

