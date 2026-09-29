"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
from model import GPT


def build_model(config):
    return GPT(config)
