Code for training transformer models.

## Objective

Incorporate evolutionary information into the training objective directly to encourage learning functional constraints rather than evolutionary memorization.

## Phylogenetic Setting

A → T1, T2

Learn fitness function / functional reasoning.

## Training Data

UniRef90 clusters.

## Current Approach

### MLM Baseline

Masked language modeling with 6% masking.

### Homolog-Conditioned Modeling

Learn P(Seq1 | Seq2), where Seq1 and Seq2 come from the same UniRef90 cluster.