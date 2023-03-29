## File Structure

- benchmarks.py
  - We use perplexity, zlib, perplexity_compare and zlib_compare to benchmark the datapoints we obtained.
  - Benchmaking results stores at gpt2/20/test_CodeGPT-small-java_checkpoint-epoch-5_victim.json
- result_analysis.ipynb
  - We analyse the benchmark result by different metrics, trying to find relevance.
  - We use the benchmark result to classify whether a datapoint is inside the trainset of our victim model, then calculate the performance of the classifier.