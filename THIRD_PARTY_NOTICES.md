# Third-party notices

`data/code/eval.json` contains problems from two public benchmarks, converted to one JSON schema (fields: `prompt`, `gold_answer`, `gold_tests`, `source`, `split`, `entry_point`).

## HumanEval (164 problems)

Source: https://github.com/openai/human-eval (Chen et al., 2021, "Evaluating Large Language Models Trained on Code"). The fields `METADATA = {'author': 'jt', ...}` inside the tests are part of the original data.

The HumanEval repository is released under the MIT License:

```
MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MBPP (257 problems)

Source: Austin et al., 2021, "Program Synthesis with Large Language Models" (https://github.com/google-research/google-research/tree/master/mbpp). The dataset is distributed under the Creative Commons Attribution 4.0 International license (https://creativecommons.org/licenses/by/4.0/). Changes made here: only the 257 problems of the test split of the sanitized MBPP release (Hugging Face dataset `mbpp`, configuration `sanitized`) are included, converted to the JSON schema above.

## Models

Generations and verdicts summarised in `tables/` come from Qwen2.5-7B-Instruct (and two GRPO-trained checkpoints of it), Llama-3.1-8B-Instruct, and the judges Qwen2.5-72B-Instruct and Qwen2.5-7B-Instruct. No model weights or raw generations are distributed here.
