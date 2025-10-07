## RAG Evaluation Results Comparison

### Cloud Results (100 samples)

| Configuration | Context Recall | Faithfulness | Factual Correctness | Answer Relevancy | Context Entity Recall | Noise Sensitivity |
|----------------|----------------|--------------|---------------------|------------------|----------------------|-------------------|
| Small chunks, no overlap, no reranking | 0.1076 | 0.6061 | 0.2990 | 0.3893 | 0.1298 | 0.0979 |
| Medium chunks, small overlap, with reranking | 0.5381 | 0.7786 | 0.4359 | 0.9217 | 0.4183 | 0.2448 |
| Larger chunks, larger overlap, reranking, and larger embedding model | 0.8052 | 0.8637 | 0.5419 | 0.9195 | 0.4482 | 0.3095 |

### Local Laptop Evaluation (70 samples)

| Configuration | Context Recall | Faithfulness | Factual Correctness | Answer Relevancy | Context Entity Recall | Noise Sensitivity |
|----------------|----------------|--------------|---------------------|------------------|----------------------|-------------------|
| Small chunks, no overlap, no reranking | 0.0523 | 0.4422 | 0.2000 | 0.7165 | 0.1919 | N/A |
| Medium chunks, small overlap, with reranking | 0.0523 | 0.4774 | 0.2390 | 0.7165 | 0.1667 | 0.0000 |
| Larger chunks, larger overlap, reranking, and larger embedding model | 0.5345 | 0.7376 | 0.4762 | 0.8988 | 0.3365 | N/A |

### Using 20b Model

| Configuration | Context Recall | Faithfulness | Factual Correctness | Answer Relevancy | Context Entity Recall | Noise Sensitivity |
|----------------|----------------|--------------|---------------------|------------------|----------------------|-------------------|
| Small chunks, no overlap, no reranking | 0.0768 | 0.3466 | 0.3350 | 0.7125 | 0.1335 | N/A |
| Medium chunks, small overlap, with reranking | 0.5623 | 0.6564 | N/A | 0.9185 | 0.4649 | N/A |

### Key Observations

- **Context Recall**: Local laptop shows significantly lower performance in the first two configurations (0.0523 vs 0.1076-0.5381), but comparable performance in the third configuration (0.5345 vs 0.8052)
- **Faithfulness**: Local laptop consistently shows lower faithfulness scores across all configurations
- **Answer Relevancy**: Interestingly, local laptop shows higher answer relevancy in simpler configurations (0.7165 vs 0.3893-0.9217)
- **Best Configuration**: For both setups, the larger chunks with overlap and reranking configuration performs best overall
