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

### Definition

Mathematically, let $\mathcal{Q} = \{q_n\}_{n=1}^N$ be a sequence of $N$ multiple choice factual questions derived from the knowledge base, each having 4 options and exactly 1 correct answer. For each question $q_n$, let $\mathcal{O}_n = \{o_n^1,o_n^2,o_n^3,o_n^4\}$ be a set of possible options and $c_n$ the corresponding correct answer.

Let $\mathcal{M}$ be a regular instruct-tuned model. We denote $a_n = \mathcal{M}(q_n) \in \{a_n^1,a_n^2,a_n^3,a_n^4\}$ as the predicted answer of the model at the $n^{th}$ question.

We can then define a model $\mathcal{M}$'s accuracy score $\mathcal{L}$ on questions $\mathcal{Q}$ as

$$
\mathcal{L}_{\mathcal{M},\mathcal{Q}} := \frac{count(q_n|a_n=c_n)}{N}
$$

Building on this, let $\mathcal{M'}$ be the instruct-tuned model that has undergone additional learning on the train split of $\mathcal{Q}$. We determine that the model successfully learned new factual knowledge if the following proposition holds:

$$
\mathcal{L}_{\mathcal{M'},\mathcal{Q}} \gt \mathcal{L}{\mathcal{M},\mathcal{Q}}
$$

In simpler terms, we say that the instruct-tuned LM $\mathcal{M'}$ has learned new information in the human sense if after undergoing further training, it gets more correct answers than its untrained version $\mathcal{M}$ on the same set of questions $\mathcal{Q}$, i.e.\ same knowledge base.

### Code

All the models can be evaluated in the notebook `eval.ipynb`.

## Results

| Task     | Model      | Base model | Fine-tuned | RAG    | Fine-tuned + RAG |
| -------- | ---------- | ---------- | ---------- | ------ | ---------------- |
| Code     | Mamba-2.8b | 0.2586     | 0.2852     | 0.2776 | 0.2700           |
|          | Gemma-2.5b | 0.3764     | 0.2877     | 0.4995 | 0.2281           |
| Research | Mamba-2.8b | 0.3117     | 0.3072     | 0.3315 | -                |
|          | Gemma-2.5b | 0.3674     | 0.1888     | 0.2961 | 0.1923           |
| Products | Mamba-2.8b | 0.3191     | 0.3547     | 0.3572 | -                |
|          | Gemma-2.5b | 0.2877     | 0.2200     | 0.4918 | 0.1924           |

## Dataset Construction

Code to construct the curriculum can be found in `/curriculum`.
