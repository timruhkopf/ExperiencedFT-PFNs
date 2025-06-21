
from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_step_lookahead import _construct_sample_weights
from botorch.sampling.normal import SobolQMCNormalSampler
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement, PosteriorMean
import torch
import numpy as np
from botorch.optim import optimize_acqf

class GaussianProcess:

    def __init__(self, acq_name="UCB"):
        #print("Using Gaussian Process as method...")

        self.acq_name = acq_name        
        #initialize acquisition functions

    def get_acquisition(self, gp = None, best_f =0.0, maximize= True):

        assert gp != None, "The model was not correctly specified"
        

        if self.acq_name == "UCB":
            return UpperConfidenceBound(gp, beta=0.1, maximize= maximize)
        
        elif self.acq_name == "EI":
            return ExpectedImprovement(gp, best_f=best_f, maximize= maximize)

        elif self.acq_name == "PM":
            return PosteriorMean(gp, maximize= maximize)

        elif self.acq_name == "PI":
            return ProbabilityOfImprovement(gp, best_f=best_f, maximize= maximize)

        elif self.acq_name == "qEI":
            sampler = SobolQMCNormalSampler(1000)
            return qExpectedImprovement(gp, best_f=best_f, sampler=sampler, maximize= maximize)
            
    def observe_and_suggest(self, X_obs, y_obs, X_pen=None,  maximize=True):

        #fit the gaussian process
        dim = X_obs.shape[1]
        X_obs = torch.tensor(X_obs, dtype=torch.double)
        y_obs = torch.tensor(y_obs, dtype=torch.double)
        
        gp = SingleTaskGP(X_obs, y_obs)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        best_f = torch.max(y_obs)  if maximize else torch.min(y_obs) 
        acq = self.get_acquisition( gp=gp, best_f=best_f, maximize=maximize)

        if X_pen is not None:
            #eval acquisition function
            X_pen = torch.tensor(X_pen, dtype=torch.double).reshape(-1,1,dim)
            eval_acq = acq( X_pen).detach().numpy()
            
            return np.argmax(eval_acq)

        else:
            dim = len(X_obs[0])
            bounds = tuple([(0,1) for _ in range(dim)])
            bounds = torch.tensor(bounds, dtype=torch.double).T
            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=20,
                options={"batch_limit": 5, "maxiter": 200},
                raw_samples=100,
            )
            # observe new values 
            new_x = candidates.detach()
            
            return new_x.numpy()




        
