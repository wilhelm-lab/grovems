from collections import Counter
import numpy as np

from .psa_utils import UtilsMixin


class AlignerMixin:
    @staticmethod
    def align(
        seq1,
        seq2,
        match=2,
        mismatch=-1,
        gap=-2,
        max_shuffle=4,
        shuffle_per_res=1.0,
        max_iso=2,
        mass_tol=0.01,
        iso_w=1.5,
        iso_len_pen=0.5,
    ):
        """One DP pass -> F, decision grid, back-pointers, path, aligned strings, match line.

        Moves: match, mismatch, gap, shuffle, and isobaric substitution decision codes:
        -1 start, 0 match, 1 mismatch, 2 gap, 3 shuffle, 4 isobaric.
        """
        n, m = len(seq1), len(seq2)
        NEG = -1e9
        F = np.full((n + 1, m + 1), NEG)
        F[0, 0] = 0
        dec = np.full((n + 1, m + 1), -1, dtype=int)
        back = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            F[i, 0] = i * gap
            dec[i, 0] = 2
            back[i][0] = (i - 1, 0)
        for j in range(1, m + 1):
            F[0, j] = j * gap
            dec[0, j] = 2
            back[0][j] = (0, j - 1)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                is_match = seq1[i - 1] == seq2[j - 1]
                best = F[i - 1, j - 1] + (match if is_match else mismatch)
                code = 0 if is_match else 1
                pred = (i - 1, j - 1)
                v = F[i - 1, j] + gap
                if v > best:
                    best, code, pred = v, 2, (i - 1, j)
                v = F[i, j - 1] + gap
                if v > best:
                    best, code, pred = v, 2, (i, j - 1)
                # shuffle: equal-multiset block, p == q
                for L in range(2, max_shuffle + 1):
                    if i >= L and j >= L:
                        w1, w2 = seq1[i - L : i], seq2[j - L : j]
                        if w1 != w2 and Counter(w1) == Counter(w2):
                            v = F[i - L, j - L] + shuffle_per_res * L
                            if v > best:
                                best, code, pred = v, 3, (i - L, j - L)
                # isobaric: equal-mass block, any p,q (covers GG<->N, AG<->Q, I<->L, ...)
                for p in range(1, max_iso + 1):
                    for q in range(1, max_iso + 1):
                        if i >= p and j >= q:
                            b1, b2 = seq1[i - p : i], seq2[j - q : j]
                            if sorted(b1) == sorted(b2):
                                continue  # identical / equal-multiset -> match or shuffle
                            m1, m2 = UtilsMixin.calculate_mass(b1), UtilsMixin.calculate_mass(b2)
                            dm = abs(m1 - m2)
                            if dm <= mass_tol:
                                v = round(
                                    F[i - p, j - q]
                                    + (iso_w * max(p, q) * (1 - dm / mass_tol) - iso_len_pen * abs(p - q))
                                )
                                if v > best:
                                    best, code, pred = v, 4, (i - p, j - q)

                F[i, j] = best
                dec[i, j] = code
                back[i][j] = pred

        # traceback from bottom-right
        cells = []
        i, j = n, m
        while (i, j) != (0, 0):
            cells.append((i, j))
            i, j = back[i][j]
        cells.append((0, 0))
        path = list(reversed(cells))

        # reconstruct aligned strings, driven by each cell's decision code
        g1, g2, mk = [], [], []
        for (i0, j0), (i1, j1) in zip(path, path[1:]):
            di, dj = i1 - i0, j1 - j0
            code = dec[i1, j1]
            if code in (0, 1):  # diagonal match / mismatch
                g1.append(seq1[i1 - 1])
                g2.append(seq2[j1 - 1])
                mk.append("|" if code == 0 else ".")
            elif code == 2:  # gap
                if di == 1 and dj == 0:
                    g1.append(seq1[i1 - 1])
                    g2.append("-")
                else:
                    g1.append("-")
                    g2.append(seq2[j1 - 1])
                mk.append("-")
            elif code == 3:  # shuffle block (di == dj)
                g1.extend(seq1[i0:i1])
                g2.extend(seq2[j0:j1])
                mk.extend("x" * di)
            else:  # code == 4 isobaric block (di=p, dj=q)
                w = max(di, dj)
                g1.extend(seq1[i0:i1].ljust(w, "-"))
                g2.extend(seq2[j0:j1].ljust(w, "-"))
                mk.extend("=" * w)

        return dict(F=F, dec=dec, back=back, path=path, g1="".join(g1), g2="".join(g2), mk="".join(mk), score=F[n, m])
