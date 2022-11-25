from __future__ import annotations
from typing import List, Tuple, Sequence, Iterable
from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
from methodtools import cached_property
from argmax_subject_to_eq import argmax_st_zero

Array = np.ndarray


class RandInterval(np.random.RandomState):
    """Uniform sampling from (a, b)"""

    def open(self, a, b):
        assert a < b
        c = a
        while c == a:
            c = a + self.random() * (b - a)
        return c


def vectorGenerator(seed=None):
    """Algorithm 1 of the paper. Asserts the 5 conditions"""
    R = RandInterval(seed)
    Q1 = R.open(0, 1 / 2)
    Q2 = R.open(1 / 2, 1)
    Q3 = R.open(1 - Q1, 1)
    P3 = R.open(1 / 2, 1)
    a = max((1 - P3) * Q1, 1 / 2 - P3 * Q3)
    b = min((1 - P3) * Q2, P3 * Q1)
    om = R.open(a, b)
    P2 = (om - Q1 * (1 - P3)) / (Q2 - Q1)
    P1 = 1 - P3 - P2
    P: Array = np.array([P1, P2, P3])
    Q: Array = np.array([Q1, Q2, Q3])
    try:
        # Verify the 5 conditions explicitly
        assert 0 < P1 < 1 and 0 < P2 < 1 and 0 < P3 < 1
        assert 0 < Q1 < 1 and 0 < Q2 < 1 and 0 < Q3 < 1
        assert P1 + P2 + P3 == 1
        assert P1 * Q1 + P2 * Q2 + P3 * Q3 > 1 / 2
        assert Q1 < 1 / 2 and Q2 > 1 / 2 and Q3 > 1 / 2
        assert Q3 + Q1 > 1
        assert P1 * Q1 + P2 * Q2 - P3 * Q1 < 0
    except AssertionError as e:
        e.args = (f"\nP={P.round(3)}\nQ={Q.round(3)}",)
        raise
    return P, Q


def arbitraryGenerator(seed=None, n=4):
    """Arbitrary generator for explanation purposes"""
    R = np.random.RandomState(seed)
    P: Array = R.random(n)  # type: ignore
    P /= P.sum()
    Q: Array = R.random(n)  # type: ignore
    return P, Q


def test_generator(many_times):
    """Test (run) the vectorGenerator many times"""
    assert many_times > 0
    for _ in range(many_times):
        vectorGenerator(seed=None)
    print(f"{many_times} tests passed")
    return


class ClockShift:
    """
    Geometric tool for measuring angles from 00h00 to 11h59
    For sorting by angle purposes, it can be shifted or set to run counter clockwise.
    """

    def __init__(self, start: float, cw: bool = True):
        self.start = start
        self.cw = cw
        self._cw = 1 if cw else -1

    def __call__(self, time: float):
        return (self._cw * (time - self.start)) % 12

    def is_first_half(self, time: float):
        return 0 <= self(time) < 6

    @staticmethod
    def coord_to_time(x: float, y: float) -> float:
        "Clock hour for ray (0,0)->(x,y), from 0.0 to 11.5999..."
        return (3 - np.arctan2(y, x) * 6 / np.pi) % 12.0


# test_generator(many_times=10**4)

import itertools
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull, convex_hull_plot_2d


class DataSource:
    """
    Class for representing a discrete and finite datasource.
    There are n possible values for x: {0, 1, ..., n-1}
    And two possible values for a: {0, 1}
    For fixed (x, a) we let i := x + a*n.
    The class contains two arrays P, Q of length 2*n:
        P[i] = Prob[X=x, A=a]
        Q[i] = Prob[Y=1 | X=x, A=a]
    """

    def __init__(self, P: Array, Q: Array, title: str = None):
        assert len(P) == len(Q)
        assert len(P) % 2 == 0
        n = len(P) // 2
        # Gradients and intra group masses:
        self.n = n
        self.P = P
        self.Q = Q
        self.nPQ = (n, P, Q)
        dx = P * (1 - 2 * Q)
        PQ = P * Q
        dy0 = PQ[:n] / np.sum(PQ[:n])
        dy1 = PQ[n:] / np.sum(PQ[n:])
        dy = np.concatenate([-dy0, dy1], axis=0)
        self._gradients = dx, dy
        self._x0 = np.sum(P * Q)
        self._bayes_sign = np.sign(self.oppDiff(self.Bayes))
        self.title = title or f"DataSource(fixed P,Q, 2n={2*self.n})"

    def print(self):
        n, P, Q = self.nPQ
        print(f"2n={2*n}")
        print(f"P={P.round(3)}")
        print(f"Q={Q.round(3)}")

    @classmethod
    def atRandom(cls, n: int, seed=None, kind="arbitrary", bias=0):
        R = np.random.RandomState(seed)
        P = R.rand(2 * n)
        P /= P.sum()
        Q = R.rand(2 * n)
        if kind == "deterministic":
            Q = (Q > 1 / 2).astype(float)
        elif kind == "problematic":
            if n == 2:
                # Generated example for Theorem "Impossibility result"
                P, Q = vectorGenerator(seed=seed)
                P = np.array([P[0], P[1], P[2] / 2, P[2] / 2])
                Q = np.array([Q[0], Q[1], Q[2], Q[2]])
            else:
                raise NotImplementedError
                # Q[A == 1] = Q[A == 1] / 2 + 1 / 2
                # Q[A == 0] = Q[A == 0] / 2 - 1 / 2
        if bias != 0:
            p = 1 + bias if bias > 0 else 1 / (1 - bias)
            Q[:n] = Q[:n] ** p
            Q[n:] = Q[n:] ** (1 / p)
        return cls(P, Q, title=f"DataSource(n={n}, seed={seed})")

    def empirical(self, n: int, seed: int = None):
        RS = np.random.RandomState(seed)
        I = RS.choice(2 * self.n, n, p=self.P)
        P = np.zeros_like(self.P)
        Q = np.zeros_like(self.Q)
        for i in I:
            P[i] += 1
            Q[i] += RS.random() < self.Q[i]
        Q = np.divide(Q, P + (Q == 0))
        P /= P.sum()
        cls = self.__class__
        title = f"{self.title}.empirical(n={n}, seed={seed})"
        return cls(P, Q, title=title)

    def err(self, R: Array):
        dx, _ = self._gradients
        assert self.is_valid(R), R
        return np.sum(R * dx) + self._x0

    def oppDiff(self, R: Array):
        _, dy = self._gradients
        assert self.is_valid(R), R
        return np.sum(R * dy)

    def is_valid(self, R):
        return all(0 <= R[i] <= 1 for i in range(2 * self.n))

    @cached_property
    def Bayes(self):
        "Classifier with maximal accuracy"
        _, _, Q = self.nPQ
        return (Q > 1 / 2).astype(float)

    @cached_property
    def Fair(self):
        "Regressor with maximal accuracy constrained to 0 bias"
        return self.pareto[-1]

    @cached_property
    def pareto(self):
        return list(self.walk_pareto_to_fair(self.Bayes))

    def XY(self, R_iter: Iterable[Array]):
        l = [(self.err(R), self.oppDiff(R)) for R in R_iter]
        return np.array(l)

    """
    @section Convex hull
    """

    def iter_hard_brute(self):
        for f in itertools.product([0, 1], repeat=2 * self.n):
            R = np.array(f)
            yield R

    def _hull_brute(self):
        hard_brute = self.XY(self.iter_hard_brute())
        hull = hard_brute[ConvexHull(hard_brute).vertices]
        return hull

    @cached_property
    def _clock_moves(self) -> List[Tuple[float, int, float]]:
        n = self.n
        dx, dy = self._gradients
        idx = [i for i in range(2 * n) if dx[i] != 0 or dy[i] != 0]
        clock = ClockShift.coord_to_time
        moves = [
            *[(clock(+dx[i], +dy[i]), i, 1) for i in idx],
            *[(clock(-dx[i], -dy[i]), i, 0) for i in idx],
        ]
        return moves

    def _sorted_clock_moves(self, cs: ClockShift):
        return sorted(self._clock_moves, key=lambda tup: cs(tup[0]))

    @cached_property
    def hull(self):
        # Start from Bayes and run clockwise
        path = []
        R = self.Bayes.copy()
        path.append(R.copy())
        cs = ClockShift(0, True)
        for _, i, v in self._sorted_clock_moves(cs):
            if R[i] != v:
                R[i] = v
                path.append(R.copy())
        return path

    """
    @section Pareto walks
    """

    def walk_pareto_to_accurate(self, R):
        """
        Walks from R to Bayes through neighboring pareto
        classifiers, yielding them on each step
        Assumes that R is a pareto classifier, possibly soft
        O(n log n)
        """
        R = R.copy()
        yield R.copy()
        cs = ClockShift(9, self._bayes_sign > 0)
        for h, i, v in self._sorted_clock_moves(cs):
            error_decrease = 6 < h < 12
            if not error_decrease:
                break
            if R[i] != v:
                R[i] = v
                yield R.copy()
        return

    def walk_pareto_to_fair(self, R):
        """
        Walks from R to Fair through neighboring pareto
        classifiers, yielding them on each step
        Assumes that R is a pareto classifier, possibly soft
        O(n log n)
        """
        _, dy = self._gradients
        R = R.copy()
        yield R.copy()
        bias = self.oppDiff(R)
        cs = ClockShift(9, self._bayes_sign < 0)
        for h, i, v in self._sorted_clock_moves(cs):
            if not cs.is_first_half(h):
                break
            if R[i] == v:
                continue
            new_bias = bias + dy[i] * (v - R[i])
            same_sign = new_bias * bias > 0
            if not same_sign:
                # Take a precise portion of last step
                v = R[i] + -bias / dy[i]
                new_bias = bias + dy[i] * (v - R[i])
                assert 0 <= v <= 1 and abs(new_bias) < 1e-9, (v, new_bias)
            R[i] = v
            yield R.copy()
            bias = new_bias
            if not same_sign or bias == 0:
                break
        return

    def the_plot(
        self,
        ds: DataSource = None,
        styles=(),
        r=0.05,
        n_random=0,
        constants=False,
        hull_border=False,
        pareto_movements=True,
        constant_movements=False,
        extra_seed=0,
    ):
        if ds is None:
            ds = self

        with plt.style.context(["grid", "scatter", *styles]):  # type:ignore
            self_is_empirical = ".empirical(" in self.title
            colors = ["tab:blue", "tab:orange"]

            if ds is not self and self_is_empirical:
                colors = colors[::-1]

            self.plot_poly(self.hull, color=colors[0], label="Feasible")
            if hull_border:
                self.plot_path(self.hull, color=colors[0])
            if ds is not self:
                label = "Real Hull" if self_is_empirical else "Empirical Hull"
                self.plot_path(ds.hull, color=colors[1], label=label)
            if constants:
                self.plot_scatter(ds.iter_const(), marker="x", label="Constant")
            # self.plot_scatter(self.iter_hard_brute(), marker=',', label='Hard')
            # self.plot_scatter([self.Q], marker='*', label='Replica')
            # self.plot_path(self.walk_zero_to_fair(), color='black')

            # p = list(self.walk_horizontally(self.rand()))
            # self.plot_path(iter(p), color='blue')
            # self.plot_movements(p[-1])

            # self.plot_path(self.walk_pareto_to_accurate(self.Fair),color='green')

            self.plot_path(ds.pareto, color="red")
            self.plot_scatter([ds.Bayes], color="black", label="Bayes")
            self.plot_scatter([ds.Fair], color="green", label="Fair")
            if pareto_movements:
                self.plot_movements(ds.Bayes)
                self.plot_movements(ds.Fair)
            if constant_movements:
                for F in ds.iter_const():
                    self.plot_movements(F)
            for i in range(n_random):
                R = np.random.RandomState(i + extra_seed)
                self.plot_movements(R.rand(2 * self.n), linewidth=0.3)

            preffix = "Apparent " if self_is_empirical else ""
            plt.xlabel(f"{preffix}Error")
            plt.ylabel(f"{preffix}Opportunity-difference")
            R_visible = [self.Bayes, self.Fair, ds.Bayes, ds.Fair]
            XX, YY = self.bounding_box(R_visible, r)
            plt.xlim(XX)
            plt.ylim(YY)
            if self is ds:
                title = self.title
            elif self_is_empirical:
                self_title = self.title.replace(ds.title, "")
                title = f"{self_title} vs {ds.title}"
            else:
                ds_title = ds.title.replace(self.title, "")
                title = f"{self.title} vs {ds_title}"
            # plt.title(title)
            plt.legend()
            plt.show()
        return

    def bounding_box(self, R_iter, r: float = 0.05):
        if r <= 0:
            xlo, xhi = (0, 1)
            ylo, yhi = (-1.05, 1.05)
        else:
            xy = self.XY(R_iter)
            x = xy[:, 0]
            y = xy[:, 1]
            xr = max(1e-3, (max(x) - min(x)) * r)
            yr = max(1e-3, (max(y) - min(y)) * r)
            xlo, xhi = (min(x) - xr, max(x) + xr)
            ylo, yhi = (min(y) - yr, max(y) + yr)
        return (xlo, xhi), (ylo, yhi)

    def plot_many_empiricals(
        self,
        n_empiricals: int,
        n_samples: int,
        seed: int = None,
        styles=(),
        r=0.05,
    ):
        with plt.style.context(["grid", "scatter", *styles]):  # type:ignore
            self.plot_poly(self.hull, color="tab:blue", label="Feasible")
            bayes_seq = []
            fair_seq = []
            RS = np.random.RandomState(seed)
            alpha = min(1, 1 / (0.1 + np.log10(n_empiricals)))
            for _ in range(n_empiricals):
                ds = self.empirical(n_samples, seed=RS.randint(0, 10**8))
                self.plot_path(ds.pareto, color="red", linewidth=0.1, alpha=alpha / 2)
                self.plot_scatter([ds.Bayes], marker="*", color="black", alpha=alpha)
                self.plot_scatter([ds.Fair], marker="*", color="green", alpha=alpha)
                bayes_seq.append(ds.Bayes)
                fair_seq.append(ds.Fair)
            bayes_XY = self.XY(bayes_seq).mean(axis=0, keepdims=True)
            fair_XY = self.XY(fair_seq).mean(axis=0, keepdims=True)
            kw = {"alpha": 1, "marker": "x", "s": 150}
            self._plot_scatter(bayes_XY, color="black", label="Bayes mean", **kw)
            self._plot_scatter(fair_XY, color="green", label="Fair mean", **kw)

            plt.xlabel(f"Error")
            plt.ylabel(f"Opportunity-difference")
            R_visible = [self.Bayes, self.Fair, *bayes_seq, *fair_seq]
            XX, YY = self.bounding_box(R_visible, r)
            plt.title(
                f"{n_empiricals} experiments of {n_samples} samples from {self.title}"
            )
            plt.xlim(XX)
            plt.ylim(YY)
            plt.legend()
            plt.show()
        return

    def plot_poly(
        self, R_iter: Iterable[Array], color="tab:blue", label=None, alpha=0.25
    ):
        poly = Polygon(self.XY(R_iter), label=label, alpha=alpha, color=color)
        plt.gca().add_patch(poly)
        # self.plot_path(self.R_iter, color='magenta')

    def plot_path(self, R_iter: Iterable[Array], color, **kwargs):
        x, y = self.XY(R_iter).T
        plt.plot(
            x,
            y,
            color=color,
            **{"alpha": 0.5, "linestyle": "-", "marker": ".", "zorder": 1.5, **kwargs},
        )

    def plot_movements(self, R: Array, color="black", alpha=0.75, linewidth=0.1):
        x0, y0 = self.XY([R])[0]
        dx, dy = self._gradients
        for i in range(2 * self.n):
            xlo = x0 - dx[i] * R[i]
            ylo = y0 - dy[i] * R[i]
            x = [xlo, xlo + dx[i]]
            y = [ylo, ylo + dy[i]]
            plt.plot(
                x,
                y,
                color=color,
                linestyle="-",
                alpha=alpha,
                linewidth=linewidth,
                marker=",",
                zorder=1.5,
            )

    def plot_scatter(self, R_iter: Iterable[Array], **kwargs):
        metrics = self.XY(R_iter)
        self._plot_scatter(metrics, **kwargs)

    def _plot_scatter(self, metrics, **kwargs):
        plt.scatter(
            metrics[:, 0],
            metrics[:, 1],
            **{"zorder": 1.6, "alpha": 0.95, "marker": "*", **kwargs},
        )

    """
    @section Additional methods
    """

    def iter_soft(self, k=11):
        dim = np.linspace(0, 1, k)
        for f in itertools.product(dim, repeat=2 * self.n):
            R = np.array(f)
            yield R

    def iter_const(self):
        yield np.zeros_like(self.P)
        yield np.ones_like(self.P)

    def optimal(self):
        dx, dy = self._gradients
        X = np.array(argmax_st_zero(-dx, np.abs(dy)))  # type:ignore
        return X

    def walk_any_to_accurate(self, X: Array):
        P = self.P
        Q = self.Q
        n = self.n
        dx, _ = self._gradients
        X = X.copy()
        yield X
        idx = [i for i in range(2 * n) if P[i] != 0]
        idx.sort(key=lambda i: -abs(dx[i]))
        for i in idx:
            if Q[i] > 1 / 2 and X[i] < 1:
                X[i] = 1
                yield X.copy()
            elif Q[i] < 1 / 2 and X[i] > 0:
                X[i] = 0
                yield X.copy()

    def walk_zero_to_fair(self):
        P = self.P
        Q = self.Q
        n = self.n
        R = np.zeros(2 * n)
        yield R.copy()

        dx, dy = self._gradients
        acc_grad = -dx
        PQ_mass = np.abs(dy)

        idx = [i for i in range(2 * n) if P[i] * Q[i] != 0 and R[i] < 1]
        idx.sort(key=lambda i: Q[i])
        pos = [i for i in idx if i >= n]
        neg = [i for i in idx if i < n]
        while pos and neg:
            i, j = pos[-1], neg[-1]
            assert Q[i] > 0 and Q[j] > 0 and R[i] < 1 and R[j] < 1

            ratio = PQ_mass[j] / PQ_mass[i]
            new_i = R[i] + (1 - R[j]) * ratio
            new_j = R[j] + (1 - R[i]) / ratio
            new_i, new_j = (1, new_j) if new_j <= 1 else (new_i, 1)
            assert new_i <= 1 and new_j <= 1

            increment = (new_i - R[i]) * acc_grad[i] + (new_j - R[j]) * acc_grad[j]
            if increment <= 0:
                break
            R[i], R[j] = (new_i, new_j)
            yield R.copy()
            if R[i] == 1:
                pos.pop()
            if R[j] == 1:
                neg.pop()
        return

    def walk_horizontally(self, R):
        R = R.copy()
        yield R.copy()
        dx, dy = self._gradients
        moves = self._clock_moves
        cs = ClockShift(9, True)
        moves.sort(key=lambda tup: -min(cs(tup[0]), 12 - cs(tup[0])))
        pos = [(i, v) for h, i, v in moves if cs.is_first_half(h)]
        neg = [(i, v) for h, i, v in moves if not cs.is_first_half(h)]
        while pos and neg:
            i, vi = pos[-1]
            j, vj = neg[-1]
            if R[i] == vi:
                pos.pop()
            elif R[j] == vj:
                neg.pop()
            else:
                top_i = dy[i] * (vi - R[i])
                top_j = -dy[j] * (vj - R[j])
                top = min(top_i, top_j)
                assert top >= 0, (top_i, top_j)
                vi = R[i] + top / dy[i]
                vj = R[j] - top / dy[j]
                if abs(vi - int(vi)) < 1e-12:
                    vi = int(vi)
                if abs(vj - int(vj)) < 1e-12:
                    vj = int(vj)
                assert 0 <= vi <= 1 and 0 <= vj <= 1, (vi, vj)
                dx_ij = dx[i] * (vi - R[i]) + dx[j] * (vj - R[j])
                if dx_ij >= 0:
                    break
                R[i], R[j] = (vi, vj)
                yield R.copy()
