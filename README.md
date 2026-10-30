# Commonsense Injection in Conversational Systems

Code for the WI-IAT 2023 paper
**"Commonsense Injection in Conversational Systems: An Adaptable Framework for Query Expansion"**
(G. Rocchietti, O. Frieder, C. I. Muntean, F. M. Nardini, R. Perego — ISTI-CNR, Pisa).

[Paper (IEEE Xplore)](https://doi.org/10.1109/WI-IAT59888.2023.00013)

## Overview

In a multi-turn conversation, a single utterance often lacks the context needed to retrieve
relevant passages. We expand each utterance with commonsense concepts related to the conversation.
The framework has two components:

- a **generator** that proposes candidate concepts using pre-trained language models and the
  ConceptNet knowledge base;
- a **selector**, a classifier trained to decide which candidate concept, if any, should be added
  to the current utterance to improve retrieval.

On TREC CAsT, the expanded utterances improve Recall@200 by up to 82% and NDCG@3 by up to 154%
over the original utterances.

## Repository structure

```
methods/
  generator_methods.py    candidate concept generation
  selector_methods.py     selector data preparation and training utilities
  pyterrier_methods.py    retrieval and evaluation with PyTerrier
  methods.py              loading of TREC CAsT topics, qrels and baselines
selector/
  selector_generate_datasets.py   builds train/test splits labelled by retrieval gain
  selector_training.py            trains the selector
  selector_evaluation.py          evaluates retrieval with the selected expansions
data/trec/
  treccast/   TREC CAsT 2019–2021 topics and manual rewrites
  qrels/      relevance judgements
  baselines/  retrieval results of the baselines
```

## Usage

Requirements: Python 3.8+, `python-terrier`, `transformers`, `ktrain`, `pandas`.
The TREC CAsT passage collection and its index are not included and must be built separately
(see the [TREC CAsT](https://www.treccast.ai/) website).
Methods that query an OpenAI model read the key from `methods/methods.py` (`api_key`): set it
there or, better, read it from an environment variable.

```bash
# 1. build selector datasets from the expansions and their retrieval results
python selector/selector_generate_datasets.py --eval_path data/trec/treccast/ \
    --qrels_path data/trec/qrels/ --baseline_path <per-query baseline results> \
    --data_path <expansions with retrieval results> --output_dir <output dir>

# 2. train the selector
python selector/selector_training.py --splits_path <output dir>/splits/ --output_dir models/

# 3. evaluate retrieval with the selected expansions
python selector/selector_evaluation.py --eval_path data/trec/treccast/ \
    --qrels_path data/trec/qrels/ --baseline_path data/trec/baselines/baselines.csv \
    --data_path <splits> --models_path models/
```

All paths default to the authors' environment; pass them explicitly as above.

## Citation

```bibtex
@inproceedings{rocchietti2023commonsense,
  author    = {Rocchietti, Guido and Frieder, Ophir and Muntean, Cristina Ioana and Nardini, Franco Maria and Perego, Raffaele},
  title     = {Commonsense Injection in Conversational Systems: An Adaptable Framework for Query Expansion},
  booktitle = {2023 IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT)},
  pages     = {48--55},
  year      = {2023},
  doi       = {10.1109/WI-IAT59888.2023.00013}
}
```

## License

MIT — see [LICENSE](LICENSE).
