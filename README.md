---
license: mit
---

# Perpetual

**Can instruct-tuned models learn new things?**

In this work we explore a novel technique inspired by human ways of learning new facts, utilizing both raw information and flashcard-style questions, attempting to teach instruct-tuned models new information without losing their conversational behavior.

We observe that Mamba-2.8b can in fact learn new factual knowledge while still retaining assistant behavior, confirming our initial hypothesis that instruct-tuned models can indeed continue to learn 🚀

## Training

A basic knowledge injection script can be done using the following:

```
python -m scripts.training.ki_model
```

Work is being done to make it extensible to more models and datasets.

## Evaluation

All the models can be evaluated in the notebook `eval.ipynb`.


## Results

| Task      | Model        | Base model | Fine-tuned | RAG    | Fine-tuned + RAG |
|-----------|--------------|------------|------------|--------|------------------|
| Code      | Mamba-2.8b   | 0.2586     | 0.2852     | 0.2776 | 0.2700           |
|           | Gemma-2.5b   | 0.3764     | 0.2877     | 0.4995 | 0.2281           |
| Research  | Mamba-2.8b   | 0.3117     | 0.3072     | 0.3315 | -                |
|           | Gemma-2.5b   | 0.3674     | 0.1888     | 0.2961 | 0.1923           |
| Products  | Mamba-2.8b   | 0.3191     | 0.3547     | 0.3572 | -                |
|           | Gemma-2.5b   | 0.2877     | 0.2200     | 0.4918 | 0.1924           |

## Dataset Construction

Code to construct the curriculum can be found in `/curriculum`.
