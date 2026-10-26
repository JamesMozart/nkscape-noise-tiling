#!/usr/bin/env python
import inkex
import random
import math
import numpy as np
from inkex import Transform


# ============ Vectorized Lightweight Simplex Noise (2D) ============
class SimplexNoise:
    def __init__(self, seed=0):
        rng = random.Random(seed)
        perm_list = list(range(256))
        rng.shuffle(perm_list)
        self.perm = np.asarray(perm_list * 2, dtype=np.int64)

        self.grad3 = np.array([
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (1, 0), (-1, 0), (0, 1), (0, -1),
        ], dtype=np.float64)

        self.F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        self.G2 = (3.0 - math.sqrt(3.0)) / 6.0

    def noise_batch(self, xin, yin):
        xin = np.asarray(xin, dtype=np.float64)
        yin = np.asarray(yin, dtype=np.float64)
        perm = self.perm
        grad3 = self.grad3
        F2, G2 = self.F2, self.G2

        s = (xin + yin) * F2
        i = np.floor(xin + s).astype(np.int64)
        j = np.floor(yin + s).astype(np.int64)
        t = (i + j) * G2
        x0 = xin - (i - t)
        y0 = yin - (j - t)

        cond = x0 > y0
        i1 = np.where(cond, 1, 0).astype(np.int64)
        j1 = np.where(cond, 0, 1).astype(np.int64)

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        ii = i & 255
        jj = j & 255

        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8

        g0x = grad3[gi0, 0]; g0y = grad3[gi0, 1]
        g1x = grad3[gi1, 0]; g1y = grad3[gi1, 1]
        g2x = grad3[gi2, 0]; g2y = grad3[gi2, 1]

        t0 = 0.5 - x0 * x0 - y0 * y0
        m0 = t0 > 0
        t0m = np.where(m0, t0 * t0, 0.0)
        n0 = np.where(m0, t0m * t0m * (g0x * x0 + g0y * y0), 0.0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        m1 = t1 > 0
        t1m = np.where(m1, t1 * t1, 0.0)
        n1 = np.where(m1, t1m * t1m * (g1x * x1 + g1y * y1), 0.0)

        t2 = 0.5 - x2 * x2 - y2 * y2
        m2 = t2 > 0
        t2m = np.where(m2, t2 * t2, 0.0)
        n2 = np.where(m2, t2m * t2m * (g2x * x2 + g2y * y2), 0.0)

        return 70.0 * (n0 + n1 + n2)

    def fbm_batch(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            total += amplitude * self.noise_batch(x * frequency, y * frequency)
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return total / max_amp if max_amp > 0 else np.zeros_like(x)

    def turbulence_batch(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            total += amplitude * np.abs(self.noise_batch(x * frequency, y * frequency))
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return (2.0 * total / max_amp - 1.0) if max_amp > 0 else np.zeros_like(x)


class NoiseTiling(inkex.EffectExtension):
    def add_arguments(self, pars):
        pars.add_argument("--width", type=float, default=300)
        pars.add_argument("--height", type=float, default=300)
        pars.add_argument("--scale_noise", type=float, default=0.02)
        pars.add_argument("--threshold", type=float, default=0.3)
        pars.add_argument("--seed", type=int, default=0)
        pars.add_argument(
            "--noise_type",
            type=str,
            default="simplex",
            choices=["simplex", "fbm", "turbulence"],
            dest="noise_type",
        )
        pars.add_argument("--octaves", type=int, default=4)
        pars.add_argument("--lacunarity", type=float, default=2.0)
        pars.add_argument("--gain", type=float, default=0.5)
        pars.add_argument("--attempts", type=int, default=500)
        pars.add_argument("--min_scale", type=float, default=0.6)
        pars.add_argument("--max_scale", type=float, default=1.4)
        pars.add_argument("--min_rotate", type=float, default=0.0)
        pars.add_argument("--max_rotate", type=float, default=360.0)
        pars.add_argument("--padding", type=float, default=2)

    def _sample_noise_batch(self, noise, nx, ny):
        kind = self.options.noise_type
        if kind == "fbm":
            return noise.fbm_batch(nx, ny,
                                   octaves=self.options.octaves,
                                   lacunarity=self.options.lacunarity,
                                   gain=self.options.gain)
        if kind == "turbulence":
            return noise.turbulence_batch(nx, ny,
                                          octaves=self.options.octaves,
                                          lacunarity=self.options.lacunarity,
                                          gain=self.options.gain)
        return noise.noise_batch(nx, ny)

    def effect(self):
        if not self.svg.selected:
            inkex.errormsg("Please select one or more objects to use as templates.")
            return

        selected_objs = list(self.svg.selected.values())

        try:
            anchor_bbox = selected_objs[0].bounding_box()
        except Exception:
            inkex.errormsg("Could not determine the bounding box of the selected object.")
            return

        area_x = anchor_bbox.left
        area_y = anchor_bbox.top
        area_w = self.options.width
        area_h = self.options.height

        # ---- Seed ----
        if self.options.seed and self.options.seed > 0:
            noise_seed = self.options.seed
        else:
            noise_seed = random.randint(0, 2147483647)

        noise_gen = SimplexNoise(seed=noise_seed)   # perm: Python random.Random
        rng = np.random.default_rng(noise_seed)     # batched draws: NumPy

        max_attempts = max(1, int(self.options.attempts))

        # ---- 1) Batch-generate candidate coordinates ----
        rand_vals = rng.random(max_attempts * 2)
        cand_x_arr = area_x + rand_vals[0::2] * area_w
        cand_y_arr = area_y + rand_vals[1::2] * area_h

        # ---- 2) Batch-compute noise ----
        scale_n = self.options.scale_noise
        noise_vals = self._sample_noise_batch(
            noise_gen,
            cand_x_arr * scale_n,
            cand_y_arr * scale_n,
        )
        prob_arr = (noise_vals + 1.0) / 2.0 - self.options.threshold

        # ---- 3) Batch acceptance test ----
        accept_rand = rng.random(max_attempts)
        accept_mask = accept_rand <= prob_arr
        accept_idx = np.nonzero(accept_mask)[0]

        # ---- 4) Pre-draw template / scale / rotation for accepted candidates ----
        n_acc = accept_idx.size
        if n_acc > 0:
            template_idx = rng.integers(0, len(selected_objs), size=n_acc)
            template_choices = [selected_objs[int(j)] for j in template_idx]

            scale_arr = rng.uniform(
                self.options.min_scale, self.options.max_scale, size=n_acc
            )
            r_min = min(self.options.min_rotate, self.options.max_rotate)
            r_max = max(self.options.min_rotate, self.options.max_rotate)
            rotate_arr = rng.uniform(r_min, r_max, size=n_acc)
        else:
            template_choices = []
            scale_arr = np.empty(0)
            rotate_arr = np.empty(0)

        # ---- 5) Placement loop (ordered insertion for collision detection) ----
        container = inkex.Group()
        container.set('id', self.svg.get_unique_id('noise_tiles'))
        parent = self.svg.get_current_layer()
        if parent is None:
            parent = self.svg.getroot()
        parent.append(container)

        template_info = []
        for t in selected_objs:
            try:
                bb = t.bounding_box()
                w = bb.width
                h = bb.height
                cx = (bb.left + bb.right) / 2.0
                cy = (bb.top + bb.bottom) / 2.0
            except Exception:
                w = h = cx = cy = None
            template_info.append((w, h, cx, cy))

        obj_index = {id(o): i for i, o in enumerate(selected_objs)}

        placed_arr = np.empty((0, 4), dtype=np.float64)
        pad = float(self.options.padding)

        for k in range(n_acc):
            idx = accept_idx[k]
            cand_x = cand_x_arr[idx]
            cand_y = cand_y_arr[idx]

            template = template_choices[k]
            ti = obj_index.get(id(template))
            if ti is None:
                continue
            orig_w, orig_h, orig_cx, orig_cy = template_info[ti]
            if orig_w is None or orig_w == 0 or orig_h == 0:
                continue

            rand_scale = scale_arr[k]
            rand_rotation = rotate_arr[k]

            diag = math.sqrt(orig_w ** 2 + orig_h ** 2)
            half_size = (diag * rand_scale) / 2.0

            cand_bbox = np.array([
                cand_x - half_size,
                cand_y - half_size,
                cand_x + half_size,
                cand_y + half_size,
            ], dtype=np.float64)

            if placed_arr.shape[0] > 0:
                bx1 = placed_arr[:, 0]
                by1 = placed_arr[:, 1]
                bx2 = placed_arr[:, 2]
                by2 = placed_arr[:, 3]
                separated = (
                    (cand_bbox[2] + pad < bx1) |
                    (cand_bbox[0] - pad > bx2) |
                    (cand_bbox[3] + pad < by1) |
                    (cand_bbox[1] - pad > by2)
                )
                if not np.all(separated):
                    continue  # collision

            tr = Transform()
            tr.add_translate(cand_x, cand_y)
            tr.add_rotate(rand_rotation)
            tr.add_scale(rand_scale)
            tr.add_translate(-orig_cx, -orig_cy)

            new_elem = template.duplicate()
            new_elem.set('id', self.svg.get_unique_id('tile'))

            existing = new_elem.get('transform')
            if existing:
                new_elem.set('transform', str(tr) + " " + existing)
            else:
                new_elem.set('transform', str(tr))

            container.append(new_elem)
            placed_arr = np.vstack([placed_arr, cand_bbox])


if __name__ == '__main__':
    NoiseTiling().run()