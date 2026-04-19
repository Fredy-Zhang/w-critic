# w-critic

`w-critic` is a WGAN-GP-based shared critic for pairwise comparison between two image generators. For each generator pair, one critic is trained with real samples and a mixed fake set built from those two generators. The trained critic is then used to evaluate which generator is closer to the real data distribution under the same reference.

![Overview of the W-Critic workflow](assets/overviews.png)

## Background

When comparing two generators, the main question is which one produces samples that are closer to the real data distribution. A critic trained to separate real images from generated images can provide a useful evaluation signal for this purpose.

In `w-critic`, the two generators are evaluated with the same critic rather than with separate critics. The critic is trained on real samples and on a pooled fake set that contains samples from both generators. This gives a shared reference for the pairwise comparison.

After training, the same critic is applied separately to withheld samples from generator A and generator B. Because both generators are measured against the same real set and the same trained critic, their distance scores can be compared directly within the pair. The generator with the smaller real-fake distance is interpreted as being closer to the real data distribution under this evaluation setting.

## Overview

The method uses one shared critic for one generator pair:

- real samples from the target dataset
- fake samples from generator A
- fake samples from generator B

The fake samples are combined into a single pooled fake distribution:

`fake_pool = fake_A ∪ fake_B`

The critic is trained to distinguish:

- `real`
- `fake_pool`

## Pairwise Evaluation Protocol

For each pair of generators:

1. Collect real samples from the target data distribution.
2. Collect fake samples from generator A and generator B.
3. Mix the two fake sets into one pooled fake distribution.
4. Train one critic on `real` versus `fake_pool`.
5. Keep separate withheld evaluation samples for generator A and generator B.
6. Apply the same trained critic to each withheld fake set against the real set.
7. Compare the resulting real-fake distances.

## Score Interpretation

The output of `w-critic` is a critic-based distance score for each generator in the pair.

- a smaller distance indicates that the generator is closer to the real data distribution under the shared critic
- a larger distance indicates that the generator is farther from the real data distribution under the shared critic

Because both generators are evaluated by the same critic, the two scores can be compared directly within the pair.

## Method Description

For each pair of generators, `w-critic` trains a single critic on a pooled fake distribution formed by mixing samples from the two generators. The same critic is then applied separately to withheld samples from each generator to obtain a pairwise comparison against the real data distribution.

## Summary

`w-critic` provides:

- one critic for one generator pair
- one pooled fake training distribution
- one shared evaluation reference for both generators

This setup supports direct pairwise comparison between two generators using the same critic.
