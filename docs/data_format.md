# Data Format

This project uses preprocessed retrieval-augmented generation (RAG) data with candidate contexts.

Due to storage limitations, the full datasets are not included in this repository.  
Instead, we provide a small set of sample files in `data_sample/` for format illustration and quick testing.

---

## Data Organization

Datasets are organized by task, and each dataset contains two splits:


data/
└── rag/
├── nq/
├── hotpotnq/
├── 2wikimultihopqa/
├── fever/
├── truthfulqa/
├── musique/
├── medical/
├── covidqa/
└── finance/


Each dataset includes:


train.json
test.json


The root data directory can be specified via configuration.

---

## Data Format

Each example consists of a query and its associated candidate contexts:

```json
{
  "query_id": "...",
  "query": "...",
  "ground_truth": ["..."],
  "candidates": [
    {
      "id": "...",
      "title": "...",
      "contents": "...",
      "score": 0.0
    }
  ]
}
Description
query denotes the input question or claim
ground_truth contains reference answers or labels
candidates represents the retrieved context pool

In our experiments, we use a fixed-size candidate pool (e.g., 10), and select top-k contexts for generation after reranking.

Sample Data

The repository provides a small number of examples under data_sample/ for:

validating input format
running lightweight sanity checks

Full Dataset Access
The complete preprocessed datasets will be released upon acceptance, subject to the licenses of the original datasets.
