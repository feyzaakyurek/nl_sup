#!/usr/bin/env python3

import torch
from tqdm import tqdm

from definitions import MisraDefinitionTask, WinodictDefinitionTask
from extrapolator import ExampleExtrapolator
from updater import NullUpdater, PromptUpdater, ExampleUpdater
from util import EvalResult

def main():
    updaters = {
        "baseline": NullUpdater(),
        #"prompter": PromptUpdater(),
        #"example_extrapolator": ExampleUpdater(ExampleExtrapolator())
    }
    #task = DefinitionTask(easy=False)
    task = MisraDefinitionTask()

    results = {k: EvalResult() for k in updaters.keys()}
    n = 0
    for supervision, targets in task:
        for updater_name, updater in updaters.items():
            model = updater(task, supervision, targets)
            with torch.no_grad():
                eval_result = task.evaluate(model, targets)
            results[updater_name].update(eval_result)
            n += 1
            del model
            torch.cuda.empty_cache()

        print()
        for updater_name, updater_results in results.items():
            print(updater_name, updater_results)

if __name__ == "__main__":
    main()
