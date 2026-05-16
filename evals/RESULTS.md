# RAG Evaluation Results

DeepEval was run across 30 document-grounded question-answer pairs from `evals/qa_pairs.json`.

Metrics:

- Relevance: DeepEval `AnswerRelevancyMetric`
- Faithfulness: DeepEval `FaithfulnessMetric` using retrieved excerpts
- Completeness: DeepEval `GEval` comparing generated answers to expected answers

## Experiment

The baseline pipeline was compared against one retrieval/indexing change:

- Baseline: chunk size `900`, overlap `150`, retrieve top `20`, final top `5`
- Variant: chunk size `450`, overlap `100`, retrieve top `20`, final top `5`

## Summary

| Run | Relevance | Faithfulness | Completeness | Overall | Source Hit Rate | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline chunk 900 | 0.9663 | 0.9875 | 0.8262 | 0.9267 | 1.0000 | 2.7430s |
| Variant chunk 450 | 0.9694 | 0.9886 | 0.8210 | 0.9263 | 1.0000 | 2.0486s |

## Conclusion

Reducing chunk size from `900` to `450` did not improve overall answer quality in this corpus. The smaller chunks slightly improved relevance and faithfulness, but completeness dropped enough that the overall score moved from `0.9267` to `0.9263`.

The variant was faster on average, so a smaller chunk size may still be useful when latency matters more than small completeness differences. For this dataset, the baseline remains the better quality default.

