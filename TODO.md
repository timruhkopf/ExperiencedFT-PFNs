# References to (re-)read:

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