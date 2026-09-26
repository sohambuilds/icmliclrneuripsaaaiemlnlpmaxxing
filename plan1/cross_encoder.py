"""Cross-encoder: a small multilingual transformer, fine-tuned on the GPU, that reads an S1 and a candidate record
together and says whether they are the same business. Its score is a second-stage feature (plan1.stage2ce).

Model: C.CE_MODEL (default paraphrase-multilingual-MiniLM-L12-v2, Apache 2.0, 118M parameters), bf16 on the GPU.
Input: "<S1 name> ; <S1 address>" and "<source> <record name> ; <record address>", using the same text as the pair
features. That text is lowercase with accents folded, Indian script in English letters, legal forms kept, dotted
forms joined and French region names dropped. So one-letter name changes, extra words, number changes and legal-form
swaps are all in view.

Training pairs come from the "ce_train" half of the extra Fit-role S1 (plan1.retrieve_extra). They are the
candidates the first-stage model scores >= S2_MIN_P1 (top S2_TOP per S1): the same kind of pairs the model scores
later. These S1 are in no other training, tuning, threshold or audit set, so the model's scores on every other set
are honest.

  python -m plan1.cross_encoder train    needs `stage2ce stage1`; -> work/plan1/<run>/stage2_ce/cross_encoder/
  python -m plan1.cross_encoder score    logits for the stage-2 sets -> stage2_ce/ce_<set>.parquet
"""
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import torch

from . import config as C
from .enrich import enriched, joined_forms
from .splits import stable_unit

S2_DIR = C.WORK_DIR / "stage2_ce"
CE_DIR = S2_DIR / "cross_encoder"
SETS = ("fit", "gbm_extra", "tune", "c_select", "dev", "fresh")
HOLDOUT_SALT = "amc26-ce-holdout-v1"


# ---------------------------------------------------------------- text
def record_text(norm: pl.DataFrame) -> pl.DataFrame:
    """entity_id, text for every record of an enriched split. S2/S3 texts start with their source."""
    name = joined_forms(pl.col("name_cons"), pl.col("country") == "France")
    addr = pl.col("addr_norm").fill_null("")
    text = pl.when(addr != "").then(name + " ; " + addr).otherwise(name).str.slice(0, 200)
    text = pl.when(pl.col("source") == "S1").then(text).otherwise(pl.col("source").str.to_lowercase() + " " + text)
    return norm.select("entity_id", text=text)


def pair_texts(pairs: pl.DataFrame, texts: pl.DataFrame) -> tuple[list[str], list[str]]:
    d = (pairs.select("s1_id", "target_id")
         .join(texts.select(s1_id="entity_id", a="text"), on="s1_id", how="left", maintain_order="left")
         .join(texts.select(target_id="entity_id", b="text"), on="target_id", how="left", maintain_order="left"))
    return d["a"].fill_null("").to_list(), d["b"].fill_null("").to_list()


# ---------------------------------------------------------------- model
def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU visible: the cross-encoder only runs on the GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def load(name_or_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(name_or_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(name_or_dir), num_labels=1)
    return tok, model.to(_device())


def _encode(tok, a: list[str], b: list[str]):
    return tok(a, b, truncation="longest_first", max_length=C.CE_MAX_LEN, padding=True, return_tensors="pt")


def _batches(tok, a: list[str], b: list[str], batches: list[np.ndarray]):
    """Tokenized batches; the next batch is tokenized on a CPU thread while the GPU works on the current one."""
    if not batches:
        return
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(_encode, tok, [a[j] for j in batches[0]], [b[j] for j in batches[0]])
        for k, idx in enumerate(batches):
            enc = nxt.result()
            if k + 1 < len(batches):
                nb = batches[k + 1]
                nxt = pool.submit(_encode, tok, [a[j] for j in nb], [b[j] for j in nb])
            yield idx, enc


@torch.inference_mode()
def score(tok, model, a: list[str], b: list[str], batch: int = C.CE_INFER_BATCH) -> np.ndarray:
    """Logits (float32) for the pairs (a[i], b[i]); pairs are sorted by length so batches carry little padding."""
    model.eval()
    dev = next(model.parameters()).device
    out = np.empty(len(a), dtype=np.float32)
    lengths = np.fromiter((len(x) + len(y) for x, y in zip(a, b)), dtype=np.int64, count=len(a))
    order = np.argsort(lengths, kind="stable")
    for idx, enc in _batches(tok, a, b, [order[i:i + batch] for i in range(0, len(a), batch)]):
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.squeeze(-1)
        out[idx] = logits.float().cpu().numpy()
    return out


# ---------------------------------------------------------------- train
def train() -> None:
    from sklearn.metrics import log_loss, roc_auc_score
    from transformers import get_linear_schedule_with_warmup

    CE_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pl.read_parquet(S2_DIR / "ce_pairs_ce_train.parquet")  # s1_id, target_id, source, label, p1
    texts = record_text(enriched("train"))
    s1 = pairs["s1_id"].unique().sort()
    hold = pl.DataFrame({"s1_id": s1.filter(pl.Series(stable_unit(s1.to_list(), HOLDOUT_SALT) < C.CE_HOLDOUT))})
    tr, ho = pairs.join(hold, on="s1_id", how="anti"), pairs.join(hold, on="s1_id", how="semi")
    a, b = pair_texts(tr, texts)
    y = tr["label"].to_numpy().astype(np.float32)
    print(f"training pairs {tr.height:,} ({y.mean():.3f} positive) from {tr['s1_id'].n_unique():,} S1; "
          f"held out {ho.height:,} pairs from {hold.height:,} S1")
    print("example pairs:")
    for i in range(3):
        print(f"  [{int(y[i])}] {a[i]}  ||  {b[i]}")

    torch.manual_seed(C.SEED)
    tok, model = load(C.CE_MODEL)
    dev = next(model.parameters()).device
    model.train()
    total = math.ceil(len(a) / C.CE_BATCH) * C.CE_EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=C.CE_LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(C.CE_WARMUP * total), total)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(C.SEED)
    t0, step = time.time(), 0
    run_sum, run_n = torch.zeros((), device=dev), 0
    for _ in range(C.CE_EPOCHS):
        order = rng.permutation(len(a))
        for idx, enc in _batches(tok, a, b, [order[i:i + C.CE_BATCH] for i in range(0, len(a), C.CE_BATCH)]):
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.squeeze(-1)
            loss = lossf(logits.float(), torch.from_numpy(y[idx]).to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_sum += loss.detach()
            run_n += 1
            if step % 500 == 0 or step == total:
                el = time.time() - t0
                print(f"  step {step:,}/{total:,}  loss {run_sum.item() / run_n:.4f}  {step / el:.1f} steps/s  "
                      f"eta {(total - step) / (step / el) / 60:.0f} min", flush=True)
                run_sum.zero_()
                run_n = 0
    train_min = (time.time() - t0) / 60
    model.save_pretrained(str(CE_DIR))
    tok.save_pretrained(str(CE_DIR))

    ha, hb = pair_texts(ho, texts)
    yh = ho["label"].to_numpy()
    p = 1 / (1 + np.exp(-score(tok, model, ha, hb).astype(np.float64)))
    res = {"logloss": float(log_loss(yh, p, labels=[0, 1])), "auc": float(roc_auc_score(yh, p)),
           "accuracy": float(((p >= 0.5) == yh).mean()), "first_stage_auc_same_pairs": float(roc_auc_score(yh, ho["p1"].to_numpy()))}
    print(f"held out: logloss {res['logloss']:.4f}, AUC {res['auc']:.5f} (first stage on the same pairs "
          f"{res['first_stage_auc_same_pairs']:.5f}), accuracy {res['accuracy']:.4f}; trained in {train_min:.0f} min")
    (CE_DIR / "ce_meta.json").write_text(json.dumps({
        "base_model": C.CE_MODEL, "max_len": C.CE_MAX_LEN, "batch": C.CE_BATCH, "lr": C.CE_LR, "epochs": C.CE_EPOCHS,
        "train_pairs": tr.height, "train_minutes": train_min, "holdout": res,
    }, indent=2), encoding="utf-8")
    print("saved:", CE_DIR)


def score_sets() -> None:
    from sklearn.metrics import roc_auc_score

    tok, model = load(CE_DIR)
    texts = record_text(enriched("train"))
    for k in SETS:
        path = S2_DIR / f"ce_{k}.parquet"
        if path.exists():
            continue
        base = pl.read_parquet(S2_DIR / f"base_{k}.parquet", columns=["s1_id", "target_id", "label"])
        t0 = time.time()
        a, b = pair_texts(base, texts)
        ce = score(tok, model, a, b)
        base.select("s1_id", "target_id").with_columns(ce=pl.Series(ce, dtype=pl.Float32)).write_parquet(path)
        secs = time.time() - t0
        print(f"{k}: {base.height:,} pairs in {secs:.0f}s ({base.height / max(secs, 1e-9):,.0f}/s), "
              f"AUC {roc_auc_score(base['label'].to_numpy(), ce):.5f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "score"):
        raise SystemExit("usage: python -m plan1.cross_encoder train | score")
    train() if sys.argv[1] == "train" else score_sets()
