# HoVer

paper: [HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification](https://arxiv.org/abs/2011.03088)

code: https://github.com/hover-nlp/hover

leaderboard: https://hover-nlp.github.io/

## Prepare data for Claim Verification + Oracle supporting facts

- clone dataset repo & download the data:
```bash
git clone https://github.com/hover-nlp/hover.git
cd hover
bash ./download_data.sh
```
- install requirements that are used for data preprocessing (from requirements.txt):
```bash
pip install tqdm stanfordcorenlp
```
- collect oracle documents
```bash
python prepare_data_for_doc_retrieval.py --data_split=dev --doc_retrieve_range=20 --oracle
python prepare_data_for_doc_retrieval.py --data_split=train --doc_retrieve_range=20 --oracle
```

These should result in ./data/hover/doc_retrieval/hover_{SPLIT_NAME}_doc_retrieval_oracle.json files with documents in context field. The oracle documents titles could be found in list of supporting facts ('supporting_facts' field).

Oracle contexts length in tokens (2-4 documents per sample):
```
min: 37, mean: 288.42, median: 274.0, max: 1215
```

All contexts length (+ retrieved by tf-idf) per sample:
```
min: 700, mean: 2402.45, median: 2362.0, max: 4857
```