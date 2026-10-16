# In-the-Wild Generalization Analysis

Evaluated on 3179 real-world utterances using checkpoint `checkpoints/rawboost_ocsoftmax_best_model.pt`.

| Metric | Score |
| :--- | :---: |
| **In-the-Wild EER** | **`34.38%`** |
| **ASVspoof 2019 LA EER** | `2.72%` |
| **Generalization Gap (Delta EER)** | `+31.66%` |
| **Accuracy (at 0.5)** | `43.69%` |
| **False Alarm Rate** | `89.08%` |
| **Miss Rate (Deepfake missed)** | `0.93%` |

### Research Takeaways:
1. **Baseline In-Domain Performance**: The model achieves high separation (2.72% EER) on controlled ASVspoof studio conditions.
2. **Out-of-Domain Vulnerability**: Real-world acoustic variations, YouTube/podcast compression artifacts, and unseen generative models widen the error rate to 34.38%.
3. **Next Technical Step**: This concrete `34.38%` baseline provides the empirical benchmark to measure the impact of **One-Class Softmax (OC-Softmax)** or domain adaptation.
