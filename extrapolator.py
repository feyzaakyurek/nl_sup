from gptcache import GPTCache

class ExampleExtrapolator:
    def __init__(self):
        self.gpt = GPTCache("gpt_cache.json")

    def __call__(self, task, supervision):
        prompt = task.make_extrapolation_prompt(supervision)
        examples = []
        i = 0
        while len(examples) < 100:
            example = self.gpt.generate(prompt, max_tokens=40, index=i)
            i += 1
            if "." in example:
                example = example[:example.index(".")+1]
            example = example.strip()
            if example in examples:
                continue
            examples.append(example)
        return examples
