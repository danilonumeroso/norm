from typing import Sequence
import random as rng


def choice(seq: Sequence):
    return rng.choice(seq)


def uniform(a: float, b: float):
    return rng.uniform(a, b)


def normal(mu: float, sigma: float):
    return rng.gauss(mu, sigma)


def integer(a: int, b: int):
    return rng.randint(a, b)


def loguniform(a: int, b: int):
    return 10 ** rng.uniform(a, b)
