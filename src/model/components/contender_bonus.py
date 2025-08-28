import torch
import matplotlib.pyplot as plt
from copy import deepcopy


def _normalize(arr: torch.Tensor) -> torch.Tensor:
    max_val = torch.max(arr)
    return arr / max_val if max_val > 0 else arr


class BudgetBasedPIBonus:
    def __init__(self, max_boost=1.0, boost_type='exp',  **boost_kwargs):
        """
        max_boost: float - maximum PI boost multiplier
        boost_type: str - 'exp', 'log', 'sigmoid', 'contender'
        boost_kwargs: dict - parameters for the boost functions
        """
        self.max_boost = max_boost
        self.boost_type = boost_type

        self.boost_kwargs = boost_kwargs

    # ---------------- Boost Functions ----------------
    @staticmethod
    def _boost_exp(b, A=1.0, alpha=5.0):
        # Exponential decay over [0,1] with stronger slope for normalized b
        raw = torch.zeros_like(b, dtype=torch.float32)
        mask = b > 0
        # For normalized fidelity b, make sure decay happens steeply in [0,1]
        raw[mask] = A * torch.exp(-alpha * b[mask])
        return _normalize(raw)

    @staticmethod
    def _boost_log(b, gamma=1.0):
        # Logarithmic decay adjusted for [0,1]
        raw = torch.zeros_like(b, dtype=torch.float32)
        mask = b > 0
        # Use log1p for numerical stability
        raw[mask] = gamma * (1 - torch.log1p(b[mask]) / torch.log(torch.tensor(2.0)))
        return _normalize(raw)

    @staticmethod
    def _boost_sigmoid(b, lambd=1.0, kappa=15.0, b0=0.001):
        # Sigmoid transition midpoint at b0 fidelity scaled for [0,1]
        raw = torch.zeros_like(b, dtype=torch.float32)
        mask = b > 0
        raw[mask] = lambd / (1.0 + torch.exp(kappa * (b[mask] - b0)))
        return _normalize(raw)

    @staticmethod
    def _boost_contender(b):
        return torch.where(b > 0, torch.tensor(1.0), torch.tensor(0.0))

    # ---------------- Main Utility ----------------
    @staticmethod
    def extract_budgets(x_train: torch.Tensor, x_test: torch.Tensor) -> torch.Tensor:
        """
        x_train, x_test shapes: (T, B, D)
        Fidelity: D=1
        Configs: D=2:D-1
        """
        T, B, D = x_test.shape
        train_fids = x_train[..., 1]      # Fidelity
        train_configs = x_train[..., 2:]  # Config features
        test_fids = x_test[..., 1]
        test_configs = x_test[..., 2:]

        flat_train_configs = train_configs.reshape(-1, train_configs.shape[-1])
        flat_train_fids = train_fids.reshape(-1)

        flat_test_configs = test_configs.reshape(-1, test_configs.shape[-1])
        flat_test_fids = test_fids.reshape(-1)

        # Convert configs to bytes keys (for exact matching)
        train_bytes = [cfg.numpy().tobytes() for cfg in flat_train_configs]
        test_bytes = [cfg.numpy().tobytes() for cfg in flat_test_configs]

        max_fidelity_map = {}
        for cfg_bytes, fid in zip(train_bytes, flat_train_fids):
            fid_val = float(fid.item())
            if cfg_bytes not in max_fidelity_map or fid_val > max_fidelity_map[cfg_bytes]:
                max_fidelity_map[cfg_bytes] = fid_val

        budgets = torch.tensor([max_fidelity_map.get(cfg_bytes, 0.0) for cfg_bytes in test_bytes])
        budgets = budgets.reshape(T, B)
        return budgets

    def compute_pi_boost(self, x_train, x_test, base_pis, advance_threshold=0.0):
        budgets = self.extract_budgets(x_train, x_test)

        func = {
            'exp': self._boost_exp,
            'log': self._boost_log,
            'sigmoid': self._boost_sigmoid,
            'is_contender': self._boost_contender
        }[self.boost_type]

        priors = func(budgets, **self.boost_kwargs)

        mask = (base_pis > advance_threshold) & (budgets > 0).flatten()
        boosted_pis = base_pis.clone()
        boosted_pis[mask] += (self.max_boost * priors).flatten()[mask]
        return torch.clamp(boosted_pis, 0., 1.)

    # ---------------- Plotting Helpers ----------------
    def plot_boost_function(self, max_budget=1, num_points=100):
        b = torch.linspace(0, max_budget, num_points)
        func = {
            'exp': self._boost_exp,
            'log': self._boost_log,
            'sigmoid': self._boost_sigmoid,
            'is_contender': self._boost_contender
        }[self.boost_type]
        boost_values = func(b, **self.boost_kwargs)

        plt.figure(figsize=(8, 5))
        plt.plot(b.numpy(), boost_values.numpy(), label=f'Boost: {self.boost_type}')
        plt.title('Budget-Based PI Boost Function')
        plt.xlabel('Invested Budget')
        plt.ylabel('Boost Value (normalized)')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_boost(self, x_train, x_test, base_pis, advance_threshold=0.0):
        budgets = self.extract_budgets(x_train, x_test)
        train = (budgets > 0).flatten()
        boosted_pis = self.compute_pi_boost(x_train, x_test, base_pis, advance_threshold)

        plt.figure(figsize=(10, 6))
        plt.hist(boosted_pis[~train].numpy(), label='Base PI (test)', alpha=0.2)
        plt.hist(base_pis[train].numpy(), label='Base PIs (train)', alpha=0.2)
        plt.hist(boosted_pis[train].numpy(), label='Boosted PIs', alpha=0.2)

        plt.title('Base vs Boosted PIs by Budget')
        plt.xlabel('PI Value')
        plt.ylabel('Count')
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------- Callable + Repr ----------------
    def __call__(self, x_train, y_train, x_test, base_pis, inc, advance_threshold=0.0):
        return self.compute_pi_boost(x_train, x_test, base_pis, advance_threshold)

    def __repr__(self):
        return (f"BudgetBasedPIBonusTorch(max_boost={self.max_boost}, "
                f"boost_type='{self.boost_type}', "
                f"boost_kwargs={self.boost_kwargs})")


if __name__ == "__main__":
    torch.manual_seed(42)
    T, B, D = 50, 1, 4

    # Random x_train and fidelities
    x_train = torch.randint(1, 10, size=(T, B, D), dtype=torch.float32)
    fidelity = torch.randint(1, 10, size=(T,))

    x_train_max = deepcopy(x_train)
    x_train_max[..., 1] = fidelity.view(T, 1)

    x_train = x_train.repeat_interleave(fidelity, dim=0)
    seqs = torch.cat([torch.arange(i) for i in fidelity]) + 2
    x_train[..., 1] = seqs.view(-1, 1).float()

    x_test = torch.randint(1, 10, size=(T+10, B, D), dtype=torch.float32)
    x_test[:T] = x_train_max
    x_test[T:, :, 1] = 1

    base_pis = torch.rand(T+10)

    bonus = BudgetBasedPIBonus(max_boost=0.05, boost_type='sigmoid')
    boosted_pis = bonus(x_train, None, x_test, base_pis, inc=None)

    print("boosted_pis:\n", boosted_pis)

    bonus.plot_boost_function()
    bonus.plot_boost(x_train, x_test, base_pis)
