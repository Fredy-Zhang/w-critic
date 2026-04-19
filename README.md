# w-critic

`w-critic` is a WGAN-GP-inspired shared critic for comparing how closely different generators match a real image distribution. Instead of training a separate critic for each generator, this project trains a single critic against a pooled fake distribution formed by mixing samples from multiple generators. The resulting critic provides a common evaluation reference for relative realism across models.

![Overview of the W-Critic workflow](assets/overviews.png)

## Motivation

A separate critic for each generator can become specialized to that generator's own artifacts. That makes cross-model comparison harder: each critic may learn a different notion of what "fake" looks like, so the resulting distances are not directly aligned.

`w-critic` addresses this by training one critic on:

- real samples from the target data distribution
- fake samples pooled from multiple generators

This shared-critic setup encourages the critic to focus on the discrepancy between the overall fake distribution and the real distribution, rather than tailoring itself to any single generator.

## Core Idea

During training, fake samples from multiple generators are mixed into one shared fake distribution:

`fake_pool = fake_A ∪ fake_B ∪ ...`

The critic is then trained to distinguish:

- `real`
- `fake_pool`

At evaluation time, the same trained critic is applied separately to withheld samples from each generator. This gives a unified way to compare generators under a single learned notion of realism.

In short:

1. Train one critic on `real` versus `pooled fake`.
2. Hold out evaluation samples for each generator.
3. Measure each generator's critic-based distance to `real`.
4. Interpret a smaller distance as closer alignment to the real distribution under the shared critic.

## Why Pooled Fake Training Helps

The main goal is not to make the critic identify which generator produced a sample. The goal is to reduce generator-specific specialization and encourage the critic to learn features associated with realness at the distribution level.

This is useful because:

- it creates a common reference for comparing multiple generators
- it reduces the risk of training one critic per generator and getting non-comparable scores
- it encourages the critic to capture shared fake-versus-real discrepancies

That said, the wording here should stay careful: mixing fake samples does not prove that the critic cannot use generator-specific cues. If different generators have distinctive artifacts, the critic may still rely on them indirectly. A better claim is that pooled training encourages the critic to focus on shared discrepancies, rather than guaranteeing complete invariance to generator identity.

## How To Interpret The Score

The distance produced by `w-critic` should be interpreted as a critic-based proxy for relative distributional closeness to real data.

Practically:

- smaller distance means the generator's outputs are judged closer to the real distribution by the shared critic
- larger distance means the outputs remain easier for the critic to separate from real samples

This makes the metric especially useful for relative comparison across generators evaluated under the same critic.

## What This Metric Is Not

`w-critic` should not be described too strongly as the true Wasserstein distance between the generator distribution and the real data distribution. In practice, the score depends on:

- the capacity of the neural critic
- the training procedure
- the sampled data
- the composition of the pooled fake set

For that reason, the score is better viewed as a learned Wasserstein-style proxy, not an exact population distance.

## Recommended Paper-Style Description

You can describe the method like this:

> We train a single critic against a pooled fake distribution formed by mixing samples from multiple generators. This shared-critic design encourages the critic to model the discrepancy between fake and real samples at the distribution level, rather than specializing to the artifacts of any individual generator. At evaluation time, the same critic is applied separately to withheld samples from each generator, and the resulting real-fake distance serves as a relative measure of closeness to the real data distribution.

Or, in a shorter form:

> By training the critic on a mixed fake distribution, we encourage it to capture the shared discrepancy between fake and real samples, providing a unified evaluator for comparing multiple generators against the real distribution.

## Limitations And Cautions

When presenting results, it helps to acknowledge a few important caveats:

- pooled training reduces but does not eliminate generator-specific bias
- score magnitudes are critic-dependent and should be compared within the same evaluation setup
- if one generator dominates the pooled fake distribution, the critic may become biased toward that generator's failure modes
- critic scaling drift or unstable training can affect comparability

## Summary

`w-critic` is best understood as a shared-critic evaluation framework:

- one critic
- one pooled fake training distribution
- one common realism standard for multiple generators

Its value is in producing a more unified and interpretable relative comparison than training a separate critic for each generator.
