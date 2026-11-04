"""Train Kuyil on a pack from pack.py with PiVo's supervised recipe. Needs only torch and model.py.

    python train.py indictts.pt runs/1.4m-s42                       # PiVo's sizes, ~585k parameters
    python train.py indictts.pt runs/5m-s42 --width 192 --ff-width 512 --text-layers 8 --decoder-layers 4

Loss: smooth-L1 on normalized features + 0.1 x squared error on log(1 + duration frames).
AdamW, lr 3e-4 with 200 warm-up updates then cosine decay, batch 8, gradient clipping at 1.
Each epoch logs to log.jsonl and saves last.pt; the lowest validation loss is kept as best.pt.
Not done yet: PiVo's 1,200-update adversarial refinement.
"""
import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import Config, Kuyil


def batches(utts, size, shuffle):
    """Random batches of similar length (sorted within shuffled pools of 50 batches) to cut padding."""
    order = list(range(len(utts)))
    if shuffle:
        random.shuffle(order)
        pool = 50 * size
        order = [j for i in range(0, len(order), pool)
                 for j in sorted(order[i : i + pool], key=lambda j: len(utts[j]["features"]))]
    groups = [order[i : i + size] for i in range(0, len(order), size)]
    if shuffle:
        random.shuffle(groups)
    for g in groups:
        yield [utts[j] for j in g]


def collate(batch, device):
    n, f = max(len(u["tokens"]) for u in batch), max(len(u["features"]) for u in batch)
    tok = torch.zeros(len(batch), n, dtype=torch.long)
    dur = torch.zeros(len(batch), n, dtype=torch.long)
    feat = torch.zeros(len(batch), f, 20)
    for i, u in enumerate(batch):
        tok[i, : len(u["tokens"])] = u["tokens"].long()
        dur[i, : len(u["durations"])] = u["durations"].long()
        feat[i, : len(u["features"])] = u["features"].float()
    voice = torch.tensor([u["voice"] for u in batch])
    lengths = torch.tensor([len(u["tokens"]) for u in batch])
    return [x.to(device) for x in (tok, dur, feat, voice, lengths)]


def losses(model, tok, dur, feat, voice, lengths):
    enc, logd, valid = model.encode(tok, voice, lengths)
    pred, fvalid = model.decode(enc, dur)
    target = (feat - model.feature_mean) / model.feature_std
    lf = F.smooth_l1_loss(pred[fvalid], target[: , : pred.shape[1]][fvalid])
    ld = F.mse_loss(logd[valid], torch.log1p(dur[valid].float()))
    return lf + 0.1 * ld, lf, ld


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("out")
    for k, v in vars(Config()).items():
        if k not in ("vocab", "voices", "feature_dim"):
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=3000, help="skip longer training utterances (PiVo: 3000)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    pack = torch.load(a.pack, weights_only=False)
    train = [u for u in pack["utterances"] if u["split"] == "train" and len(u["features"]) <= a.max_frames]
    val = [u for u in pack["utterances"] if u["split"] == "val"]
    cfg = Config(voices=len(pack["voices"]), **{k: getattr(a, k) for k in vars(Config()) if hasattr(a, k)})
    model = Kuyil(cfg).to(device)
    model.feature_mean.copy_(pack["feature_mean"]); model.feature_std.copy_(pack["feature_std"])
    params = sum(p.numel() for p in model.parameters())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    steps_total = a.max_steps or a.epochs * math.ceil(len(train) / a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps_total)))))
    print(f"{device}, {params:,} parameters, {len(train)} train / {len(val)} val utterances, {steps_total} steps", flush=True)

    step, best, t0 = 0, float("inf"), time.time()
    log = open(out / "log.jsonl", "a")
    for epoch in range(a.epochs):
        model.train()
        tot = n = 0
        for b in batches(train, a.batch, True):
            loss, lf, ld = losses(model, *collate(b, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tot += loss.item(); n += 1
            if a.max_steps and step >= a.max_steps:
                break
        model.eval()
        with torch.no_grad():
            vl = [torch.stack(losses(model, *collate(b, device))).cpu() for b in batches(val, a.batch, False)]
        vloss, vf, vd = torch.stack(vl).mean(0).tolist()
        rec = {"epoch": epoch, "step": step, "train": tot / max(n, 1), "val": vloss, "val_features": vf,
               "val_durations": vd, "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        ck = {"config": model.config(), "state_dict": model.state_dict(), "symbols": pack["symbols"],
              "voices": pack["voices"], "epoch": epoch, "step": step, "val": vloss, "params": params, "args": vars(a)}
        torch.save(ck, out / "last.pt")
        if vloss < best:
            best = vloss
            torch.save(ck, out / "best.pt")
        if a.max_steps and step >= a.max_steps:
            break
