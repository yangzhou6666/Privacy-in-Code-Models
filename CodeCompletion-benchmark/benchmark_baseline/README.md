## File Structure

- benchmarks.py
  - We use perplexity, zlib, perplexity_compare, and zlib_compare to benchmark the data points we obtained.
  - Benchmarking results are stored at gpt2/20/test_CodeGPT-small-java_checkpoint-epoch-5_victim.json
- result_analysis.ipynb
  - We analyze the benchmark results using different metrics, trying to find relevance.
  - We use the benchmark results to classify whether a data point is inside the training set of our victim model, then calculate the performance of the classifier.
