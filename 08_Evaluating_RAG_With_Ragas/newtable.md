## Results of the three runs with 100 samples

| Configuration | Context Recall | Faithfulness | Factual Correctness | Answer Relevancy | Context Entity Recall | Noise Sensitivity |
|----------------|----------------|--------------|---------------------|------------------|----------------------|-------------------|
| Small chunks, no overlap, no reranking | 0.1076 | 0.6061 | 0.2990 | 0.3893 | 0.1298 | 0.0979 |
| Medium chunks, small overlap, with reranking | 0.5381 | 0.7786 | 0.4359 | 0.9217 | 0.4183 | 0.2448 |
| Larger chunks, larger overlap, reranking, and larger embedding model | 0.8052 | 0.8637 | 0.5419 | 0.9195 | 0.4482 | 0.3095 |