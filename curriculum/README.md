# Curriculum Construction

## Quick Start

1. OpenAI API KEY

   ```
   export OPENAI_API_KEY=sk-XXXX
   ```

2. Construct curriculum

   ```
   python text-splitter.py
   ```

## Overview

We explore fine-tuning 3 different types of data:

1. Structured data like those from a company database.
2. Raw text like news articles and research papers.
3. Codebases from publicly available hosted code repositories.

For each section above, we start with the following:

1. `wdc/products-2017`:

- The Web Data Commons Training and Test Sets for Large-Scale Product Matching contain product offers from different e-shops
- Available from 🤗 Huggingface Datasets.

2. `Jamba: A Hybrid Transformer-Mamba Language Model`

- Published on 03/28/24, this paper outlines the groundbreaking Jamba model.
- Available from [ArXiv](https://arxiv.org/pdf/2403.19887.pdf).

3. `🔥 flamethrower`

- A random *pet project* open-sourced on GitHub with <10,000 lines of code.
- Available on GitHub at [scottsus/flamethrower](https://github.com/scottsus/flamethrower).
