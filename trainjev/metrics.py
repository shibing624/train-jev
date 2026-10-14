"""Distribution-level and decision-level metrics against soft targets."""
import numpy as np


def metrics(rows, probs):
    """Per-kind KL, top-1 (decisive rows only), ECE, noul AUROC and score MAE, plus overall KL."""
    out, by = {}, {}
    for r, p in zip(rows, probs):
        by.setdefault(r["kind"], []).append((r, p))
    allkl = []
    for kind, pairs in by.items():
        kl, top1, ev_err, conf, corr, au_y, au_s = [], [], [], [], [], [], []
        for r, p in pairs:
            t = np.asarray(r["target"], float)
            q = np.clip(np.asarray(p, float), 1e-12, 1)
            kl.append(float((t * (np.log(np.clip(t, 1e-12, 1)) - np.log(q))).sum()))
            decisive = t.max() - np.sort(t)[-2] > 1e-9
            if decisive:
                top1.append(int(q.argmax() == t.argmax()))
                conf.append(q.max())
                corr.append(int(q.argmax() == t.argmax()))
            if kind == "score":
                lv = np.arange(len(t))
                ev_err.append(abs((q * lv).sum() - (t * lv).sum()))
            if kind == "noul" and decisive:
                au_y.append(int(t[1] > t[0]))
                au_s.append(q[1])
        d = {"n": len(pairs), "kl": float(np.mean(kl)), "top1": float(np.mean(top1)) if top1 else None}
        if ev_err:
            d["score_mae"] = float(np.mean(ev_err))
        if au_y and 0 < sum(au_y) < len(au_y):
            from sklearn.metrics import roc_auc_score

            d["auroc"] = float(roc_auc_score(au_y, au_s))
        if conf:
            d["ece"] = ece(conf, corr)
        out[kind] = d
        allkl += kl
    out["all"] = {"n": len(rows), "kl": float(sum(allkl) / max(1, len(allkl)))}
    return out


def ece(conf, correct, bins=10):
    """Expected calibration error with equal-width confidence bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins) if b else (conf <= 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)
