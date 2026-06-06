"""
Markov Chain 模型 — N-gram 转移概率
速度最快，可解释性最强，作为 baseline
"""
import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Optional


class MarkovModel:
    """
    N 阶 Markov Chain（N-gram 语言模型）
    给定过去 N 个 token → 预测下一个 token 的概率分布
    """

    def __init__(self, order: int = 3, vocab_size: int = None, smoothing: float = 1e-6):
        self.order    = order
        self.vocab_size = vocab_size
        self.smoothing  = smoothing
        # counts[(t1,t2,...,tN)] = {next_token: count}
        self.counts: dict = defaultdict(lambda: defaultdict(float))
        self._trained = False

    # ── 训练 ──────────────────────────────────────────
    def fit(self, sequences: list[list[int]]) -> "MarkovModel":
        """
        sequences: 多个 token index 序列（每只股票一条序列）
        """
        for seq in sequences:
            for i in range(len(seq) - self.order):
                context = tuple(seq[i : i + self.order])
                next_t  = seq[i + self.order]
                self.counts[context][next_t] += 1.0
        self._trained = True
        return self

    # ── 预测 ──────────────────────────────────────────
    def predict_proba(self, context: list[int]) -> np.ndarray:
        """
        context: 最近 order 个 token index
        返回 shape [vocab_size] 的概率数组
        """
        assert self.vocab_size, "vocab_size 未设置"
        key = tuple(context[-self.order:])
        raw = self.counts.get(key, {})

        probs = np.full(self.vocab_size, self.smoothing)
        for idx, cnt in raw.items():
            probs[idx] += cnt
        probs /= probs.sum()
        return probs

    def predict(self, context: list[int]) -> tuple[int, float]:
        """返回 (best_token_idx, confidence)"""
        probs = self.predict_proba(context)
        best  = int(np.argmax(probs))
        return best, float(probs[best])

    # ── 评估 ──────────────────────────────────────────
    def accuracy(self, sequences: list[list[int]]) -> float:
        correct, total = 0, 0
        for seq in sequences:
            for i in range(len(seq) - self.order):
                context  = seq[i : i + self.order]
                true_next = seq[i + self.order]
                pred, _  = self.predict(context)
                if pred == true_next:
                    correct += 1
                total += 1
        return correct / total if total else 0.0

    def perplexity(self, sequences: list[list[int]]) -> float:
        log_sum, total = 0.0, 0
        for seq in sequences:
            for i in range(len(seq) - self.order):
                context  = seq[i : i + self.order]
                true_next = seq[i + self.order]
                probs = self.predict_proba(context)
                log_sum += np.log(probs[true_next] + 1e-12)
                total   += 1
        return float(np.exp(-log_sum / total)) if total else float("inf")

    # ── 持久化 ────────────────────────────────────────
    def save(self, path: str = "saved_models/markov.json"):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        data = {
            "order": self.order,
            "vocab_size": self.vocab_size,
            "smoothing": self.smoothing,
            # counts key 是 tuple，序列化为 str
            "counts": {
                str(k): dict(v) for k, v in self.counts.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[markov] saved → {path}")

    @classmethod
    def load(cls, path: str = "saved_models/markov.json") -> "MarkovModel":
        with open(path) as f:
            data = json.load(f)
        m = cls(order=data["order"], vocab_size=data["vocab_size"],
                smoothing=data["smoothing"])
        for k_str, v in data["counts"].items():
            # 把 "(1, 2, 3)" 还原为 tuple
            k = tuple(int(x.strip()) for x in k_str.strip("()").split(",") if x.strip())
            m.counts[k] = defaultdict(float, {int(ki): float(vi) for ki, vi in v.items()})
        m._trained = True
        print(f"[markov] loaded ← {path}")
        return m
