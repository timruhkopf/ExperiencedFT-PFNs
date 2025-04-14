# TOODs: 

- [ ] Find out why the distillation on same task does not work yet.

### References to (re-)read:

- [x] [Statistical foundations of PFNs](https://arxiv.org/abs/2305.11097)
- [ ] [PFN4HPO](https://arxiv.org/pdf/2305.17535)
- [ ] TabPFN
- [x] [In-Context Data Distillation with TabPFN](https://arxiv.org/abs/2402.06971)
- [ ] Scaling TabPFN: Sketching and feature selection for tabular PFNs
- [ ] [ifBO](https://arxiv.org/pdf/2404.16795)
- [ ] [A Closer Look at In-Context Learning under Distribution Shifts](https://arxiv.org/pdf/2305.16704)
- [ ] [Exchangeable Sequence Models Quantify Uncertainty Over Latent Concepts](https://arxiv.org/pdf/2408.03307)  De Finetti's predictive probabilistic reasoning for sequence models
- [ ] Calibration in deep learning a survey of the sota
- [ ] [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) --> maybe we can make use of smart reasoning in the latent space to emulate chain of thought here and help the acquisition function 
- [ ] [A Closer Look at TabPFN v2: Strength, Limitation, and Extension](https://arxiv.org/pdf/2502.17361) randomized feature tokens are critical to tabpfn v2's success to unify heterogeneous datasets,  leaveone-fold-out approach, transforming TabPFN v2
- [ ] [](https://arxiv.org/pdf/2405.16156)  Sparse Mixture of In-Context Prompters: We solve PFN’s alignment limitations with “Context-Aware PFN” (CAPFN),
- [ ] [Attic](https://openreview.net/pdf?id=DSl9sSuUhp) tabpfns outperform trees like xgboost,  Attic assigns one token to each feature of every observation "cell tokens".
In our intuition, this is because the cell-token architecture is
feature-order invariant

### BASELINE
- [ ] [](https://arxiv.org/pdf/2405.16156) 
  which finetunes PFNS for downstream datasets via bootstrapping.
  --> Context-Aware Finetuning (CAPFN): misalignment of prior with real data --> use dataset bootstrapped subsets from the larger datasets to fine tune adapters -- i.e. bootstrap mechanism becomes the prior
  so we can use the previous datasets as bootstrapped fine-tuning

### skimmable: 
- [ ] [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564) row, column attention --> then fixed size embedding for scale
  into a feature extractor and revealing its capability to simplify data distributions, divide-and-conquer mechanism inspired by Chain-of-Thought prompting for scale
- [ ] [](https://openreview.net/pdf?id=f8aganC0tN) linear attention, SSM for scale
- [ ] [TABDPT: SCALING TABULAR FOUNDATION MODELS](https://arxiv.org/pdf/2410.18164?) real data + prior  training, random row as target for representation learning
- [ ] [TABPFN Unleashed](https://arxiv.org/pdf/2502.02527?) e BETA (Bagging and Encoder-based
Fine-tuning for TabPFN Adaptation)  minimize both bias and variance. To reduce bias, we introduce a lightweight encoder to better align downstream tasks with the pre-trained TabPFN
- [ ] [Tabforest](https://arxiv.org/pdf/2405.13396) use tabpfn prior + rf prior to fine tune.

# Open Questions
A. Small acquisitions for a curve will unlikely bring much of a benefit, as the horizon is relatively small and the information gained is likely small. 
Prompting and optimizing the acquisition every step of the way is costly and brings little benefit -- instead, we could encourage "self play"; i.e. sample and 
rollout for multiple steps (check how the Acquisition MFPI random from ifbo does it). So we could potentially annotate a few points based on their likely outcome and 
"overconfidently" attempt to sample some new points.--> i.e. query for new points, sample them like from the resulting y distribution and then rollout under that condition (instead of evaluating them)
    
    We apply multiple path decoding with a sampling
    temperature T > 0 for generating m reasoning paths and answers { r i 1 , r i 2 , . . . , r i m } for each ques- tion x i in Dtrain , and use majority voting (self-consistency) to select the most consistent, highest confidence answer (Wang et al., 2022b).
    Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou.  Self-consistency improves chain of thought reasoning in language models. ArXiv, abs/2203.11171, 2022b.



# Important notes: 
The ifbo prior `ifbo.priors.ftpfn_prior.MLP` has a peculiarity, I wasn't aware of when reading the paper: 

```python
import torch 

class MLP(torch.nn.Module):
    # [...]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.linears[:-1]:
            x = linear(x)
            # NOTICE: no two fwd calls with the same hp on the same MLP instantiation will be the
            # same even before the outputs are interpreted as parameters to the base learning curves
            x = x + torch.randn_like(x) * self.preactivation_noise_std
            x = torch.tanh(x)
        x = self.linears[-1](x)
        return x + torch.randn_like(x) * self.output_noise
```

part of the prior is actually also dealing with different levels of fidelities: `ifbo.priors.ftpfn_prior.get_batch`
```python
# determine the number of fidelity levels (ranging from 1: BB, up to seq_len)
n_levels = int(np.round(10 ** np.random.uniform(0, 3)))
```


This is basically the dirichlet prior: 
```python

alpha = 10 ** np.random.uniform(-4, -1)
weights = np.random.gamma(alpha, alpha, seq_len) + EPS
p = weights / np.sum(weights)


```
Gamma sampling converts to Dirichlet distribution via normalization
Smaller α → sparser allocations (focus on few configs):  Exploitation-focused (low α →
few configs get most tokens)
Larger α → uniform distribution (equal attention): Exploration-friendly 




