"""Training objective: KL(teacher || model) over the question's options, plus an ordinal RPS term for ``score``."""
import torch
import torch.nn.functional as F


def kl_rps(z, target, kind, rps_w):
    """KL(target || softmax(z)) (+ ``rps_w`` * ranked probability score for score rows). ``z``, ``target``: ``[n]``."""
    logp = F.log_softmax(z, -1)
    t = target.clamp_min(0)
    kl = (t * (torch.log(t.clamp_min(1e-12)) - logp)).sum()
    if kind == "score" and rps_w:
        cp, ct = torch.cumsum(logp.exp(), 0), torch.cumsum(t, 0)
        kl = kl + rps_w * ((cp - ct) ** 2)[:-1].sum() / (len(t) - 1)
    return kl
