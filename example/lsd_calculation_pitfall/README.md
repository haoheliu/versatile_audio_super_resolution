# Log Spectral Distance (LSD) Scaling Effect Demonstration

This file demonstrates the importance of energy scaling when calculating Log Spectral Distance (LSD) between audio signals before and after super-resolution operation. The demonstration shows how normalizing the energy levels of the lower frequency bands can significantly impact the accuracy and fairness of LSD measurements.

## Overview

Log Spectral Distance is a commonly used metric for evaluating audio quality and similarity. However, when comparing audio signals with different energy levels, the raw LSD calculation can be biased by these energy differences rather than focusing on the actual spectral characteristics.

This file demonstrate the implementation of a **frequency energy unification** approach that scales the super-resolution output audio's lower frequency energy to match the ground truth, ensuring a fair comparison based on spectral shape rather than overall energy levels.

## Key Concepts

### What is LSD?
Log Spectral Distance measures the difference between two audio signals' spectral envelopes. It's calculated as:

```
LSD = sqrt(mean((log10(target² / (estimate² + ε)) + ε)²))
```

**Important Note**: Even for the same audio signal x, the LSD between x and x*a (where a is a scaling factor) is not 0.0. This is because LSD is sensitive to energy differences, not just spectral shape differences.

### Why Energy Scaling Matters
- **Eliminates Energy Bias**: Removes systematic differences due to recording conditions or processing
- **Focuses on Spectral Shape**: Compares the actual spectral characteristics rather than energy levels
- **Improves Fairness**: Provides more meaningful comparisons for audio quality assessment

### The Scaling Approach
We scale the target spectrogram so that its lower 10% frequency energy matches the ground truth (**because we assume the lower frequencies information does not need to change before and after super-resolution operation**):

```python
scaling_factor = gt_lower_energy / target_lower_energy
scaled_target = target_spectrogram * scaling_factor
```

## Results

### Audio Files Used
- **Ground Truth**: `gt_2.wav` (reference audio)
- **Target**: `selected_2.wav` (audio to be evaluated)

### Energy Analysis
```
GT lower frequency energy: 0.273710
Target lower frequency energy: 0.166209
Scaling factor applied: 1.646787
```

### LSD Comparison

| Metric | Without Scaling | With Scaling | Improvement |
|--------|----------------|--------------|-------------|
| Average LSD | 1.5963 | 1.3623 | 0.2339 |

**Overall improvement: 14.66%**

The results clearly demonstrate the significant changes achieved through energy scaling. 

## Conclusion

Energy scaling before LSD calculation is important for fair and accurate audio quality assessment. 

The implemented approach scales the target audio's lower frequency energy to match the ground truth, ensuring that LSD measurements focus on spectral shape differences rather than energy level differences.
