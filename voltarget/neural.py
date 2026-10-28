"""LSTM volatility forecaster in PyTorch.

Question this answers: does a neural network beat the classical models
(GARCH, HAR) at forecasting tomorrow's volatility, under the same rules?

Design choices, each made to keep the comparison fair:
  * Same convention as forecasters.py: forecast[t] uses only returns before t.
  * Walk-forward: retrained once a year (every 252 days) on all data so far.
    Training is slower than fitting GARCH, so we refit less often; the README
    states this, since it slightly handicaps the LSTM.
  * Input: the last 22 daily returns and their absolute values (one month).
  * Output: log of tomorrow's variance. Predicting the log keeps the forecast
    positive without clipping.
  * Loss: QLIKE, the same loss used to score every forecaster. Minimizing it is
    the same as maximizing a Gaussian likelihood, so the network is trained on
    exactly the objective it is judged by.
  * Early stopping on the most recent 20% of each training window, which also
    respects time order (we validate on later data, never earlier).
  * Fixed seeds, so results reproduce exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEQ_LEN = 22
REFIT_EVERY = 252
MIN_TRAIN = 500


def _windows(x: np.ndarray, end: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs for targets seq_len..end-1: each row is the seq_len returns before it."""
    idx = np.arange(seq_len, end)
    X = np.stack([x[i - seq_len : i] for i in idx])
    feats = np.stack([X, np.abs(X)], axis=-1)        # (samples, seq_len, 2)
    return feats, x[idx]


def lstm_vol(
    r: pd.Series,
    refit_every: int = REFIT_EVERY,
    min_train: int = MIN_TRAIN,
    hidden: int = 16,
    max_epochs: int = 60,
    patience: int = 6,
    seed: int = 0,
    anchored: bool = False,
) -> pd.Series:
    """anchored=False: plain LSTM. Inputs are raw returns and the network
    predicts the variance level itself.

    anchored=True: hybrid. The EWMA forecast sets the level and the network
    only learns a correction to it:
        log sigma2[t] = log ewma2[t] + network(returns / ewma_sigma[t])
    Dividing returns by the current EWMA volatility makes the inputs scale-free,
    so a crash looks like "returns 3x bigger than recent normal" rather than a
    number the network never saw in training. This fixes the plain LSTM's
    failure in 2008, when it was trained on calm years and could not forecast
    volatility far above anything in its training data.
    """
    import torch
    from torch import nn

    from .forecasters import ewma_vol

    torch.set_num_threads(1)
    x = r.to_numpy() * 100.0                          # percent returns: better-scaled for training
    n = len(x)
    var = np.full(n, np.nan)
    ewma2 = (ewma_vol(r).to_numpy() * 100.0) ** 2     # ewma2[t] uses returns before t only
    anchor_log = np.log(np.maximum(ewma2, 1e-12)) if anchored else np.zeros(n)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, seq):                       # seq: (batch, seq_len, 2)
            out, _ = self.lstm(seq)
            return self.head(out[:, -1, :]).squeeze(-1)   # log variance

    def qlike(log_s2, y):
        return (log_s2 + y**2 * torch.exp(-log_s2)).mean()

    for k in range(min_train, n, refit_every):
        torch.manual_seed(seed + k)
        rng = np.random.default_rng(seed + k)

        X, y = _windows(x, k, SEQ_LEN)                # targets strictly before day k
        targets = np.arange(SEQ_LEN, k)
        if anchored:
            X = X / np.sqrt(ewma2[targets])[:, None, None]       # scale-free inputs
        scale = np.abs(X).reshape(-1, 2).std(axis=0) + 1e-8      # normalize with training data only
        X = X / scale
        base = anchor_log[targets]
        split = int(len(X) * 0.8)
        T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        Xtr, ytr, btr = T(X[:split]), T(y[:split]), T(base[:split])
        Xva, yva, bva = T(X[split:]), T(y[split:]), T(base[split:])

        net = Net()
        with torch.no_grad():
            # Start at the anchor (correction = 0) or, for the plain model, at average variance.
            start = 0.0 if anchored else float(np.log(np.mean(y[:split] ** 2)))
            net.head.bias.fill_(start)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        best, best_state, bad = np.inf, None, 0
        for _ in range(max_epochs):
            net.train()
            order = rng.permutation(len(Xtr))
            for b in range(0, len(order), 256):
                batch = order[b : b + 256]
                opt.zero_grad()
                loss = qlike(btr[batch] + net(Xtr[batch]), ytr[batch])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = qlike(bva + net(Xva), yva).item()
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {name: t.clone() for name, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:          # None only if no epoch ran
            net.load_state_dict(best_state)

        # Forecast the next block, each day from the 22 returns before it.
        stop = min(k + refit_every, n)
        days = np.arange(k, stop)
        Xf = np.stack([x[t - SEQ_LEN : t] for t in days])
        Xf = np.stack([Xf, np.abs(Xf)], axis=-1)
        if anchored:
            Xf = Xf / np.sqrt(ewma2[days])[:, None, None]
        Xf = Xf / scale
        net.eval()
        with torch.no_grad():
            log_s2 = anchor_log[days] + net(T(Xf)).numpy()
        var[k:stop] = np.exp(log_s2)

    return pd.Series(np.sqrt(var) / 100.0, index=r.index, name="lstm_anchored" if anchored else "lstm")


def lstm_anchored_vol(r: pd.Series, **kwargs) -> pd.Series:
    return lstm_vol(r, anchored=True, **kwargs)
