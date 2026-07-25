# CIPHER Fine-Tuning and Inference

The manuscript describes a separate reconstruction-based fine-tuning and risk-scoring pipeline:

1. Initialize from the self-supervised CIPHER foundation-model checkpoint.
2. Fine-tune reconstruction using internal non-ICI-P cases and L1 loss.
3. Reconstruct each baseline CT within the lung mask.
4. Calculate voxel-level reconstruction error.
5. Aggregate reconstruction error to a patient-level CIPHER score.
6. Select and lock the classification threshold using internal development data.
7. Apply the locked model and threshold to held-out and external cohorts without retraining.

Executable fine-tuning, inference, lung-mask, score-aggregation, and threshold-selection scripts were not included in the uploaded source bundle. Add those verified scripts here before public release.
