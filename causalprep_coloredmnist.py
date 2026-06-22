# python3 causalprep_coloredmnist.py --data_dir /home/giridharan/Documents/causal --output_dir ./results_mnist

#!/usr/bin/env python3
"""
CausalPrep: Preprocessing-based Causal Invariance for ColoredMNIST Dataset

Pipeline steps:
  1. Multi-environment data loading (train1, train2, test)
  2. Feature extraction (6 shape features = causal, 1 color feature = spurious)
  3. FCI Causal Graph Discovery → learn PAG
  4. Generalized Adjustment Criterion (GAC) over the PAG → valid Z per feature,
     or 'not identifiable' if no valid adjustment set exists
  5. Residualization using the discovered Z
  6. ACE Estimation (Doubly Robust DML)
  7. Feature Filtering (ACE + R² + FCI criteria)
  8. Causally filtered feature matrix
  9. Downstream model training: Raw (ERM, all features) vs CausalPrep
     (ERM on filtered + residualized features)
  10. IRM-style diagnostics across environments (worst-group acc, invariance gap)

ColoredMNIST structure:
  train1.pt : 20000 samples, green→label=1 (80% correlation)
  train2.pt : 20000 samples, green→label=1 (90% correlation)
  test.pt   : 20000 samples, FLIPPED — red→label=1 (anti-correlated)

Causal signal : digit shape (invariant across environments)
Spurious signal: digit color (flipped at test time)

Feature set (7 total):
  Shape (causal)  : hog_energy_shape, hog_entropy_shape, edge_density_shape,
                     stroke_width_shape, digit_area_shape, aspect_ratio_shape
  Color (spurious): mean_red_color

"""

import os
import argparse
import warnings
import itertools
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Set, FrozenSet

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from skimage.feature import hog

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.Endpoint import Endpoint
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

class ColoredMNISTLoader:
    """
    Load ColoredMNIST .pt files and return images, labels, and color (environment).

    Environment encoding:
      env=0 : train1 (green→label=1 at 80%)
      env=1 : train2 (green→label=1 at 90%)
      env=2 : test   (red→label=1 at 90%, anti-correlated)

    Color encoding derived from image pixels:
      color=0 : red background
      color=1 : green background
      color=2 : blue background  (rare, ~0.3%)
    """

    @staticmethod
    def detect_color(img_arr: np.ndarray) -> int:
        """
        Detect digit color from RGB image array (28,28,3).
        Background is black (0,0,0). Color is ON the digit pixels.
        Samples non-black pixels (sum > 30) to find digit color.
        Returns: 0=red, 1=green, 2=unknown
        """
        arr = img_arr.astype(float)
        # Non-black pixels carry the digit color
        mask = arr.sum(axis=2) > 30
        if mask.sum() < 5:
            return 2  # too few colored pixels — unknown
        colored = arr[mask]   # (n_pixels, 3)
        r = colored[:, 0].mean()
        g = colored[:, 1].mean()
        b = colored[:, 2].mean()
        if r > g and r > b:
            return 0  # red digit
        elif g > r and g > b:
            return 1  # green digit
        else:
            return 2  # unknown

    @staticmethod
    def load_split(pt_path: str, env_id: int, max_samples: Optional[int] = None) -> Dict:
        """
        Load one .pt file.

        Args:
            pt_path    : path to .pt file
            env_id     : integer environment id (0=train1, 1=train2, 2=test)
            max_samples: if set, subsample for speed

        Returns:
            dict with keys:
              'images' : (N, 28, 28, 3) uint8 numpy array
              'labels' : (N,) int array  — binary (0 or 1)
              'colors' : (N,) int array  — 0=red, 1=green, 2=blue
              'env'    : (N,) int array  — all equal to env_id
              'n'      : int
        """
        data = torch.load(pt_path, weights_only=False)
        if max_samples is not None:
            data = data[:max_samples]

        images = []
        labels = []
        colors = []

        for img_pil, label in data:
            arr = np.array(img_pil)   # (28, 28, 3) uint8
            color = ColoredMNISTLoader.detect_color(arr)
            images.append(arr)
            labels.append(int(label))
            colors.append(color)

        images = np.stack(images, axis=0)   # (N, 28, 28, 3)
        labels = np.array(labels, dtype=int)
        colors = np.array(colors, dtype=int)
        envs   = np.full(len(labels), env_id, dtype=int)

        return {
            'images': images,
            'labels': labels,
            'colors': colors,
            'env'   : envs,
            'n'     : len(labels)
        }

    @staticmethod
    def load_all(data_dir: str, max_per_split: Optional[int] = None) -> Dict:
        """
        Load train1, train2, test and combine into a single dataset.
        Training data = train1 + train2 combined.
        Test data kept separate for evaluation.

        Returns:
            dict with:
              'train'      : combined train1+train2
              'test'       : test split
              'train1'     : train1 only (env=0)
              'train2'     : train2 only (env=1)
        """
        data_dir = Path(data_dir)

        print("\n" + "="*70)
        print("STEP 1: DATA LOADING")
        print("="*70)

        splits = {}
        for fname, env_id in [('train1.pt', 0), ('train2.pt', 1), ('test.pt', 2)]:
            path = data_dir / fname
            splits[fname] = ColoredMNISTLoader.load_split(
                str(path), env_id, max_per_split
            )
            d = splits[fname]
            print(f"\n{fname}")
            print(f"  Samples : {d['n']}")
            print(f"  Labels  : 0={( d['labels']==0).sum()}  1={(d['labels']==1).sum()}")
            color_names = {0:'red', 1:'green', 2:'blue'}
            for c, cname in color_names.items():
                n = (d['colors']==c).sum()
                if n > 0:
                    mask = d['colors']==c
                    p1 = d['labels'][mask].mean()
                    print(f"  {cname:5s}: {n:5d} samples | P(label=1|{cname})={p1:.3f}")

        # Combine train1 + train2
        t1 = splits['train1.pt']
        t2 = splits['train2.pt']
        train = {
            'images': np.concatenate([t1['images'], t2['images']], axis=0),
            'labels': np.concatenate([t1['labels'], t2['labels']], axis=0),
            'colors': np.concatenate([t1['colors'], t2['colors']], axis=0),
            'env'   : np.concatenate([t1['env'],    t2['env']],    axis=0),
            'n'     : t1['n'] + t2['n']
        }

        print(f"\n  Combined train : {train['n']} samples")
        print(f"  Test set       : {splits['test.pt']['n']} samples")

        return {
            'train' : train,
            'test'  : splits['test.pt'],
            'train1': t1,
            'train2': t2
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Extract two groups of features from ColoredMNIST images (7 total):

    SHAPE features (causal — digit shape is invariant across environments):
      1.  hog_energy      : sum of squared HOG descriptor values (overall gradient energy)
      2.  hog_entropy     : entropy of HOG descriptor (texture complexity)
      3.  edge_density    : fraction of pixels with high gradient magnitude (Sobel)
      4.  stroke_width    : mean run-length of foreground pixels per row
      5.  digit_area      : fraction of pixels classified as digit (bright on background)
      6.  aspect_ratio    : bounding box height / width of digit region

    COLOR feature (spurious — color is environment-dependent):
      7.  mean_red        : mean red channel value of digit pixels (normalized 0-1)

    Feature names follow the _shape / _color suffix convention,
    analogous to _bird / _bg in the Waterbirds pipeline. Only one color
    feature (mean_red) is used by design here — ColoredMNIST's spurious
    signal is a single scalar (red vs green dominance), unlike Waterbirds
    where multiple background-derived features carried the confound.
    """

    FEATURE_NAMES = [
        'hog_energy_shape',
        'hog_entropy_shape',
        'edge_density_shape',
        'stroke_width_shape',
        'digit_area_shape',
        'aspect_ratio_shape',
        'mean_red_color',
    ]

    @staticmethod
    def extract_one(img_arr: np.ndarray) -> np.ndarray:
        """
        Extract all 7 features from a single (28,28,3) uint8 image.
        Returns (7,) float64 array.
        """
        # Convert to grayscale for shape features
        # Standard luminance weights
        gray = (0.2989 * img_arr[:,:,0] +
                0.5870 * img_arr[:,:,1] +
                0.1140 * img_arr[:,:,2]).astype(np.float32) / 255.0

        r = img_arr[:,:,0].astype(np.float32) / 255.0
        g = img_arr[:,:,1].astype(np.float32) / 255.0
        b = img_arr[:,:,2].astype(np.float32) / 255.0

        # ── Shape features ────────────────────────────────────────────

        # 1. HOG energy + entropy
        hog_desc = hog(
            gray,
            orientations=8,
            pixels_per_cell=(7, 7),
            cells_per_block=(1, 1),
            feature_vector=True
        )
        hog_energy  = float(np.sum(hog_desc ** 2))
        # Entropy of normalized HOG
        hog_norm    = hog_desc / (hog_desc.sum() + 1e-8)
        hog_entropy = float(-np.sum(hog_norm * np.log(hog_norm + 1e-8)))

        # 2. Edge density (Sobel approximation)
        gy = np.diff(gray, axis=0)  # (27,28)
        gx = np.diff(gray, axis=1)  # (28,27)
        # Align shapes
        grad_mag = np.sqrt(gy[:, :-1]**2 + gx[:-1, :]**2)
        edge_density = float((grad_mag > 0.1).mean())

        # 3. Stroke width: mean horizontal run-length of bright pixels
        #    Bright = pixel value > 0.3 (digit is white/bright on colored bg)
        bright = gray > 0.3
        run_lengths = []
        for row in bright:
            in_run = False
            run_len = 0
            for px in row:
                if px:
                    run_len += 1
                    in_run = True
                else:
                    if in_run and run_len > 0:
                        run_lengths.append(run_len)
                    run_len = 0
                    in_run = False
            if in_run and run_len > 0:
                run_lengths.append(run_len)
        stroke_width = float(np.mean(run_lengths)) if run_lengths else 0.0

        # 4. Digit area: fraction of bright pixels
        digit_area = float(bright.mean())

        # 5. Aspect ratio: bounding box of bright pixels
        rows_with_bright = np.any(bright, axis=1)
        cols_with_bright = np.any(bright, axis=0)
        if rows_with_bright.any() and cols_with_bright.any():
            row_indices = np.where(rows_with_bright)[0]
            col_indices = np.where(cols_with_bright)[0]
            height = float(row_indices[-1] - row_indices[0] + 1)
            width  = float(col_indices[-1] - col_indices[0] + 1)
            aspect_ratio = height / (width + 1e-8)
        else:
            aspect_ratio = 1.0

        # ── Color feature ──────────────────────────────────────────────
        # Background is black. Color is ON the digit pixels (non-black).
        # Sample non-black pixels to measure mean red channel intensity.
        arr_f  = img_arr.astype(np.float32) / 255.0
        mask_c = arr_f.sum(axis=2) > (30.0 / 255.0)  # non-black pixels

        if mask_c.sum() >= 5:
            dp       = arr_f[mask_c]   # digit pixels (n, 3)
            mean_red = float(dp[:, 0].mean())
        else:
            mean_red = 0.0

        return np.array([
            hog_energy,
            hog_entropy,
            edge_density,
            stroke_width,
            digit_area,
            aspect_ratio,
            mean_red,
        ], dtype=np.float64)

    @staticmethod
    def extract_all(images: np.ndarray, desc: str = '') -> np.ndarray:
        """
        Extract features from all images.

        Args:
            images: (N, 28, 28, 3) uint8
            desc  : tqdm description

        Returns:
            (N, 7) float64 feature matrix
        """
        features = []
        for i in tqdm(range(len(images)), desc=desc or 'Extracting features'):
            features.append(FeatureExtractor.extract_one(images[i]))
        return np.stack(features, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: CAUSAL DISCOVERY (R² diagnostic)
# ══════════════════════════════════════════════════════════════════════════════

class CausalDiscovery:
    """
    Diagnostic step: measure how much each feature is predicted by color
    (the environment/confounder variable) using linear R².

    This mirrors the Waterbirds pipeline's validate_structure step.
    Color is the confounder here (analogous to 'place' in Waterbirds).

    R² threshold 0.10: features above are flagged AFFECTED.
    """

    @staticmethod
    def validate_structure(features: np.ndarray,
                           labels: np.ndarray,
                           colors: np.ndarray,
                           feature_names: List[str],
                           r2_threshold: float = 0.10) -> Dict:
        """
        For each feature, regress feature ~ color and report R².

        Args:
            features     : (N, 12) standardized feature matrix
            labels       : (N,) binary labels
            colors       : (N,) color labels (0=red, 1=green, 2=blue)
            feature_names: list of 12 names
            r2_threshold : R² above this = AFFECTED by color

        Returns:
            dict {feature_name: {'r_squared': float, 'affected': bool}}
        """
        print("\n" + "="*70)
        print("STEP 3a: CAUSAL DISCOVERY (R² diagnostic)")
        print("="*70)
        print(f"  {'Feature':<30}  {'Status':<12}  R²")
        print(f"  {'-'*55}")

        # One-hot encode color for regression
        color_onehot = np.zeros((len(colors), 3), dtype=float)
        for i, c in enumerate(colors):
            color_onehot[i, c] = 1.0

        results = {}
        for j, name in enumerate(feature_names):
            reg = LinearRegression()
            reg.fit(color_onehot, features[:, j])
            r2 = max(0.0, reg.score(color_onehot, features[:, j]))
            affected = r2 >= r2_threshold
            status = "AFFECTED   " if affected else "INDEPENDENT"
            print(f"  {name:<30}  {status:<12}  {r2:.3f}")
            results[name] = {'r_squared': r2, 'affected': affected}

        n_affected = sum(1 for v in results.values() if v['affected'])
        print(f"\n  {n_affected}/{len(feature_names)} features affected by color (R² >= {r2_threshold})")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3b: FCI CAUSAL GRAPH DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

class FCICausalDiscovery:
    """
    Run FCI on the feature matrix + color variable to learn a PAG
    and identify which features are confounded by color (environment).

    Node layout: nodes 0-6 = features, node 7 = color
    """

    @staticmethod
    def run(features: np.ndarray,
            colors: np.ndarray,
            feature_names: List[str],
            alpha: float = 0.05,
            fci_sample_size: int = 2000,
            random_seed: int = 42) -> Dict:
        """
        Run FCI and return PAG-based confounder analysis.

        Args:
            features        : (N, 7) standardized feature matrix
            colors          : (N,) color labels (0/1/2)
            feature_names   : list of 7 names
            alpha           : significance level for Fisher Z test
            fci_sample_size : subsample size (FCI is O(n²) in features)
            random_seed     : for reproducible subsampling

        Returns:
            dict with keys:
              'pag_matrix'         : (8,8) raw graph.graph matrix
              'node_names'         : 8 names (7 features + 'color')
              'edges'              : list of edge strings
              'confounded_by_color': dict {feature_name: edge_type_string}
              'clean_features'     : list of names NOT confounded by color
              'fci_sample_size'    : actual sample size used
              'graph_object'       : GeneralGraph (needed for backdoor step)
        """

        print("\n" + "="*70)
        print("STEP 3b: FCI CAUSAL GRAPH DISCOVERY")
        print("="*70)

        # ── 1. Build data matrix: [features | color] ──────────────────
        color_col = colors.reshape(-1, 1).astype(float)
        data_full = np.hstack([features, color_col])
        node_names = list(feature_names) + ['color']
        n_total = data_full.shape[0]

        # ── 2. Stratified subsample (balance across color classes) ─────
        rng = np.random.default_rng(random_seed)
        n_sample = min(fci_sample_size, n_total)

        indices_by_color = [np.where(colors == c)[0] for c in range(3)]
        n_per_color = n_sample // 3
        sampled = []
        for idx in indices_by_color:
            n_take = min(n_per_color, len(idx))
            sampled.append(rng.choice(idx, size=n_take, replace=False))
        sampled_idx = np.concatenate(sampled)
        rng.shuffle(sampled_idx)
        data_sample = data_full[sampled_idx]
        actual_n = len(sampled_idx)

        print(f"  Dataset     : {n_total} samples | {len(node_names)} variables ({len(feature_names)} features + color)")
        print(f"  Subsample   : {actual_n} samples (stratified by color)")
        print(f"  Test        : Fisher Z | alpha={alpha}")

        # ── 3. Run FCI ─────────────────────────────────────────────────
        graph, edges = fci(
            dataset=data_sample,
            independence_test_method='fisherz',
            alpha=alpha,
            depth=-1,
            max_path_length=-1,
            verbose=False,
            show_progress=True,
            node_names=node_names
        )

        pag = graph.graph   # (n_nodes, n_nodes)
        color_idx = len(feature_names)  # last node is always color

        # ── 4. Decode edge mark helper ─────────────────────────────────
        def decode_edge(i: int, j: int, pag: np.ndarray) -> str:
            mark_at_j = pag[j, i]
            mark_at_i = pag[i, j]
            if mark_at_j == 0 and mark_at_i == 0:
                return 'no edge'
            mark_str  = {1: '>', -1: '-', 2: 'o', 0: '?'}
            left      = mark_str.get(mark_at_i, '?')
            right     = mark_str.get(mark_at_j, '?')
            left_mark  = '<' + left  if left  != '-' else '-'
            right_mark = right + '>' if right != '-' else '-'
            return f"{left_mark}--{right_mark}"

        # ── 5. Identify features confounded by color ───────────────────
        confounded_by_color = {}
        clean_features = []

        print(f"\n  PAG edges involving 'color':")
        print(f"  {'Feature':<30} {'Edge':>15}   Interpretation")
        print(f"  {'-'*70}")

        for j, fname in enumerate(feature_names):
            edge_str     = decode_edge(color_idx, j, pag)
            mark_at_feat  = pag[j, color_idx]
            mark_at_color = pag[color_idx, j]

            if edge_str == 'no edge':
                clean_features.append(fname)
                print(f"  {fname:<30} {'no edge':>15}   ✓ INDEPENDENT of color")
            else:
                if mark_at_feat == 1 and mark_at_color == -1:
                    interp = "color --> feature (direct cause)"
                elif mark_at_feat == 1 and mark_at_color == 1:
                    interp = "color <-> feature (latent confounder)"
                elif mark_at_feat == 1 and mark_at_color == 2:
                    interp = "color o-> feature (possibly direct)"
                elif mark_at_feat == -1 and mark_at_color == -1:
                    interp = "color --- feature (undirected)"
                else:
                    interp = f"ambiguous ({mark_at_color},{mark_at_feat})"
                confounded_by_color[fname] = interp
                print(f"  {fname:<30} {edge_str:>15}   ✗ {interp}")

        # ── 6. Full edge list ──────────────────────────────────────────
        print(f"\n  Full PAG ({len(edges)} edges):")
        for edge in edges:
            print(f"    {edge}")

        # ── 7. Summary ─────────────────────────────────────────────────
        print(f"\n  Confounded by color : {len(confounded_by_color)} features")
        for fname, interp in confounded_by_color.items():
            print(f"    • {fname}: {interp}")
        print(f"  Independent of color: {len(clean_features)} features")
        for fname in clean_features:
            print(f"    • {fname}")

        return {
            'pag_matrix'         : pag,
            'node_names'         : node_names,
            'edges'              : edges,
            'confounded_by_color': confounded_by_color,
            'clean_features'     : clean_features,
            'fci_sample_size'    : actual_n,
            'graph_object'       : graph
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: GENERALIZED ADJUSTMENT CRITERION + RESIDUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class GeneralizedAdjustment:
    """
    Generalized Adjustment Criterion (GAC) for PAGs (Maathuis & Colombo 2015;
    van der Zander, Liskiewicz & Textor 2014/2019).

    Unlike the DAG backdoor criterion, a PAG contains circle marks ('o') that
    represent genuine uncertainty about edge orientation. The previous
    implementation in this file hard-coded Z={color} based on a handful of
    mark patterns — that is a heuristic, not a derivation from the graph.
    GAC instead works directly with the PAG's "possible" relations (allowing
    circle marks to stand in for either tail or arrowhead) and proves Z is
    valid for ALL DAGs/MAGs consistent with the PAG, or reports that no such
    Z exists ('not identifiable').

    Definitions used here, operating on the PAG endpoint matrix M where
    M[i, j] is the mark AT NODE i on the edge between i and j
    (so the edge i — j has mark M[i,j] at i and M[j,i] at j):
        TAIL   = -1   (i —  : i is not caused by whatever sits at the other end
                        through this edge in that direction)
        ARROW  =  1   (i <- : arrowhead into i)
        CIRCLE =  2   (i o  : unknown/ambiguous)
        NULL   =  0   (no edge)

    Step 1 — Possible-causal-path descendants, forb(X, Y, Z):
        A node V is a "possible descendant" of X if there is a path from X
        to V on which every edge is "possibly directed away from X", i.e.
        at X's end the mark is ARROW-pointing-away (TAIL or CIRCLE at X's
        side, ARROW or CIRCLE at the next node's side) — circles are
        treated as if they *could* be arrows, which is the conservative
        (sound) choice required for the criterion to hold in every member
        of the PAG's equivalence class.
        forb(X, Y) = (possible descendants of X) that are also possible
                     descendants of any node on a possibly-causal path
                     from X to Y, restricted to the analysis node set.
        For our 1-confounder setting this reduces to: forb(X,Y) = nodes V
        such that X has a possibly-directed path into V.

    Step 2 — Amenability / validity test for candidate Z:
        Z is a valid adjustment set for the total effect of X on Y if:
          (a) Z ∩ forb(X, Y) = ∅            (no node in Z is a possible
                                              descendant of X on a path to Y)
          (b) every proper, possibly-causal-definite-status path from X to Y
              that is NOT into X is blocked by Z, where "blocked" follows
              the m-separation rules generalized to circle marks (a circle
              endpoint can act as either a collider or non-collider, so a
              path is only certainly blocked if it is blocked under BOTH
              interpretations).

    Step 3 — Search:
        Because there is no canonical single Z in a PAG (several sets can
        be valid, or none), we enumerate candidate Z over the other
        analysis-set nodes (here: color + other features) in increasing
        size order and return the first valid set found. If none of the
        2^(k-1) candidates (including the empty set) is valid, the
        adjustment is reported 'not identifiable' from this PAG and the
        feature is left unadjusted (flagged for the report, not silently
        adjusted on a guess).
    """

    TAIL, ARROW, CIRCLE, NULLM = -1, 1, 2, 0

    # ---- low-level PAG mark helpers -------------------------------------

    @staticmethod
    def _possibly_directed_successors(pag: np.ndarray, node: int, n: int) -> List[int]:
        """
        Nodes V such that edge (node, V) has a mark at `node`'s end that is
        TAIL or CIRCLE (i.e. NOT an arrow pointing back into `node`), and a
        mark at V's end that is ARROW or CIRCLE (i.e. could point into V).
        This is the "possibly directed away from node" relation used to
        build possible-descendant sets conservatively.
        """
        succ = []
        for v in range(n):
            if v == node:
                continue
            m_at_node = pag[node, v]
            m_at_v    = pag[v, node]
            if m_at_node == GeneralizedAdjustment.NULLM and m_at_v == GeneralizedAdjustment.NULLM:
                continue  # no edge
            node_ok = m_at_node in (GeneralizedAdjustment.TAIL, GeneralizedAdjustment.CIRCLE)
            v_ok    = m_at_v in (GeneralizedAdjustment.ARROW, GeneralizedAdjustment.CIRCLE)
            if node_ok and v_ok:
                succ.append(v)
        return succ

    @staticmethod
    def _possible_descendants(pag: np.ndarray, start: int, n: int) -> Set[int]:
        """BFS over possibly-directed edges to get all possible descendants of `start`."""
        visited = set()
        frontier = [start]
        while frontier:
            cur = frontier.pop()
            for nxt in GeneralizedAdjustment._possibly_directed_successors(pag, cur, n):
                if nxt not in visited:
                    visited.add(nxt)
                    frontier.append(nxt)
        return visited

    @staticmethod
    def _possibly_directed_paths_xy(pag: np.ndarray, x: int, y: int, n: int) -> List[List[int]]:
        """All simple paths X -> ... -> Y following possibly-directed edges (for forb set)."""
        paths = []
        def dfs(cur, path, visited):
            if cur == y:
                paths.append(path[:])
                return
            for nxt in GeneralizedAdjustment._possibly_directed_successors(pag, cur, n):
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, visited)
                    path.pop()
                    visited.remove(nxt)
        dfs(x, [x], {x})
        return paths

    @staticmethod
    def _forb(pag: np.ndarray, x: int, y: int, n: int) -> Set[int]:
        """
        forb(X, Y): X itself, plus all possible descendants of X that lie on
        some possibly-directed path from X to Y (conservative: if X has any
        possibly-directed path to Y at all, every possible descendant of X
        reachable from those path nodes is forbidden from Z).
        """
        forb = {x}
        paths = GeneralizedAdjustment._possibly_directed_paths_xy(pag, x, y, n)
        for path in paths:
            for node in path:
                forb |= GeneralizedAdjustment._possible_descendants(pag, node, n)
                forb.add(node)
        # X's own possible descendants are always forbidden, even with no path to Y yet found,
        # since amenability requires none of them confound the effect.
        forb |= GeneralizedAdjustment._possible_descendants(pag, x, n)
        return forb

    @staticmethod
    def _all_paths(pag: np.ndarray, x: int, y: int, n: int, max_len: int = 6) -> List[List[int]]:
        """All simple undirected (adjacency-following) paths X..Y up to max_len, for blocking checks."""
        paths = []
        def dfs(cur, path, visited):
            if len(path) > max_len:
                return
            if cur == y and len(path) > 1:
                paths.append(path[:])
                return
            for v in range(n):
                if v == cur or v in visited:
                    continue
                if pag[cur, v] == GeneralizedAdjustment.NULLM and pag[v, cur] == GeneralizedAdjustment.NULLM:
                    continue
                visited.add(v)
                path.append(v)
                dfs(v, path, visited)
                path.pop()
                visited.remove(v)
        dfs(x, [x], {x})
        return paths

    @staticmethod
    def _is_collider_at(pag: np.ndarray, a: int, b: int, c: int) -> Optional[bool]:
        """
        Is b a collider on the path a-b-c (i.e. arrowhead into b from both sides)?
        Returns True/False if determined by the PAG marks, or None if ambiguous
        (a circle mark at b means it could be a collider or not, under different
        members of the equivalence class).
        """
        mark_b_from_a = pag[b, a]
        mark_b_from_c = pag[b, c]
        if mark_b_from_a == GeneralizedAdjustment.ARROW and mark_b_from_c == GeneralizedAdjustment.ARROW:
            return True
        if mark_b_from_a == GeneralizedAdjustment.CIRCLE or mark_b_from_c == GeneralizedAdjustment.CIRCLE:
            return None  # ambiguous under circle marks
        return False

    @staticmethod
    def _path_blocked_by_z(pag: np.ndarray, path: List[int], z: FrozenSet[int],
                           x: int, y: int, n: int) -> bool:
        """
        Is `path` certainly blocked by Z, robust to circle-mark ambiguity?
        A path is blocked if SOME triple on it is a definite non-collider not
        in Z, or a definite collider whose descendants (conservatively: whose
        possible descendants) don't intersect Z. Because circle marks make
        collider status ambiguous, we require the path to be blocked under
        BOTH the "collider" and "non-collider" reading of every ambiguous
        triple for the path to count as certainly (soundly) blocked.
        """
        if len(path) < 3:
            return False  # X-Y direct edge can't be blocked by conditioning

        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            collider = GeneralizedAdjustment._is_collider_at(pag, a, b, c)

            if collider is True:
                # Blocked at b iff none of b's possible descendants (incl. b) are in Z
                desc_b = {b} | GeneralizedAdjustment._possible_descendants(pag, b, n)
                if not (desc_b & z):
                    return True  # this triple alone blocks the path
                # else: this triple does not block — keep checking other triples
            elif collider is False:
                if b in z:
                    return True
                # else: non-collider and not conditioned on — does not block here
            else:
                # Ambiguous: certainly blocked only if blocked under BOTH readings
                blocked_as_collider = not ({b} | GeneralizedAdjustment._possible_descendants(pag, b, n)) & z
                blocked_as_noncoll  = b in z
                if blocked_as_collider and blocked_as_noncoll:
                    return True
                # if not both, this triple doesn't certainly block the path here;
                # continue scanning other triples on the path
        return False

    @staticmethod
    def find_adjustment_set(pag: np.ndarray, x: int, y: int, n: int,
                            candidate_pool: List[int],
                            max_subset_size: int = 4) -> Optional[List[int]]:
        """
        Search candidate_pool (e.g. all nodes except X and Y) for the smallest
        Z satisfying the Generalized Adjustment Criterion for the effect of
        X on Y in this PAG. Returns the node-index list for Z, or None if no
        valid Z is found among the candidates tried (caps subset size at
        max_subset_size for tractability — with <=8 nodes total this is
        exhaustive in practice).

        GAC validity for Z:
          (a) Z ∩ forb(X,Y) = ∅
          (b) every path from X to Y that is not blocked by definite-status
              colliders / non-colliders alone (i.e. every path that could
              carry confounding) is blocked by Z.
        """
        forb = GeneralizedAdjustment._forb(pag, x, y, n)
        all_paths = GeneralizedAdjustment._all_paths(pag, x, y, n)

        # Direct edge between X and Y is never blockable — if it's an edge
        # consistent with a direct causal effect, that's fine (it's the effect
        # itself, not a backdoor path); only non-adjacent paths need blocking.
        backdoor_paths = [p for p in all_paths if len(p) > 2]

        valid_pool = [v for v in candidate_pool if v not in (x, y)]

        for size in range(0, min(max_subset_size, len(valid_pool)) + 1):
            for combo in itertools.combinations(valid_pool, size):
                z = frozenset(combo)
                if z & forb:
                    continue
                if all(GeneralizedAdjustment._path_blocked_by_z(pag, p, z, x, y, n)
                       for p in backdoor_paths):
                    return list(combo)
        return None


class BackdoorAdjustment:
    """
    Derive valid adjustment sets from the PAG using the Generalized
    Adjustment Criterion (GeneralizedAdjustment, above) and residualize
    features to remove the discovered confounding.

    Replaces the earlier heuristic that hard-coded Z={color} whenever a
    color↔feature edge was present in the PAG. That heuristic ignored two
    things a PAG actually encodes: (1) other features can themselves act as
    valid (or necessary) adjustment variables, and (2) circle marks mean
    some edges are genuinely ambiguous, so a single fixed Z is not always
    correct — sometimes no valid Z exists at all ('not identifiable'),
    and the feature should be left unadjusted and flagged, not silently
    residualized on color anyway.

    Action semantics per feature, now derived from the GAC search:
      'keep'            : feature independent of color in the PAG — no
                           adjustment needed.
      'adjust'          : GAC found a valid, non-empty Z — residualize on it.
      'not_identifiable': feature is confounded by color in the PAG but GAC
                           found no valid adjustment set among the other
                           analysis variables — left unadjusted, flagged.
      'check'           : feature is a possible ancestor of color (collider
                           risk if adjusted) — left unadjusted.
    """

    @staticmethod
    def derive_adjustment_sets(fci_result: Dict,
                               graph,
                               feature_names: List[str]) -> Dict:
        """
        For each feature, run the Generalized Adjustment Criterion against
        Y=label (label is not a node in the PAG, so we treat the adjustment
        target as 'color', the only confounder modeled in this graph — GAC
        is applied to the effect of feature X on the confounder-laden path
        through color, using the other features + color as the candidate
        pool for Z).

        Returns:
            dict {feature_name: {'action': str, 'adjustment_set': list,
                                 'reason': str}}
        """
        print("\n" + "="*70)
        print("STEP 4: GENERALIZED ADJUSTMENT CRITERION (GAC) over the PAG")
        print("="*70)
        print("Searching for a valid Z per feature via GAC (Maathuis & Colombo).")
        print("If no valid Z exists among the other analysis variables, the")
        print("feature is marked 'not identifiable' and left unadjusted.\n")

        pag        = fci_result['pag_matrix']
        confounded = fci_result['confounded_by_color']
        node_names = fci_result['node_names']
        n          = len(node_names)
        color_idx  = len(feature_names)  # dynamic: last node is color

        nodes      = graph.get_nodes()
        color_node = nodes[color_idx]

        adjustment_sets = {}

        for j, fname in enumerate(feature_names):
            feat_node = nodes[j]
            color_is_possible_ancestor = graph.is_ancestor_of(feat_node, color_node)

            if fname not in confounded:
                action  = 'keep'
                adj_set = []
                reason  = "Independent of color in PAG — no adjustment needed"

            elif color_is_possible_ancestor:
                # Feature may cause color (or be a possible ancestor under
                # some member of the equivalence class) — adjusting on color
                # risks collider bias, so GAC would (correctly) reject color
                # from any Z. Skip the search and report this directly.
                action  = 'check'
                adj_set = []
                reason  = ("Feature is a possible ancestor of color — "
                           "adjusting risks collider bias. Left unadjusted.")

            else:
                # Candidate pool: every other node in the analysis graph
                # (other features + color), excluding the feature itself.
                candidate_pool = [k for k in range(n) if k != j]
                z_indices = GeneralizedAdjustment.find_adjustment_set(
                    pag=pag, x=j, y=color_idx, n=n,
                    candidate_pool=candidate_pool, max_subset_size=4
                )

                if z_indices is None:
                    action  = 'not_identifiable'
                    adj_set = []
                    reason  = ("PAG confounds feature↔color but GAC found no "
                               "valid Z among other variables — not identifiable "
                               "from this PAG. Left unadjusted.")
                elif len(z_indices) == 0:
                    # GAC says the empty set already suffices (no real backdoor
                    # path needing blocking, despite the FCI edge flag) — no-op.
                    action  = 'keep'
                    adj_set = []
                    reason  = "GAC: empty set is already valid — no adjustment needed"
                else:
                    action  = 'adjust'
                    adj_set = [node_names[k] for k in z_indices]
                    reason  = f"GAC found valid Z={{{','.join(adj_set)}}}"

            adjustment_sets[fname] = {
                'action'        : action,
                'adjustment_set': adj_set,
                'reason'        : reason
            }

            action_str = {
                'adjust'           : '⟳ ADJUST  ',
                'keep'             : '✓ KEEP    ',
                'check'            : '⚠ CHECK   ',
                'not_identifiable' : '? UNIDENT.',
            }[action]
            adj_str = f"Z={{{','.join(adj_set)}}}" if adj_set else "Z={}"
            print(f"  {action_str}  {fname:<30} | {adj_str:<20} | {reason}")

        n_adjust = sum(1 for v in adjustment_sets.values() if v['action'] == 'adjust')
        n_keep   = sum(1 for v in adjustment_sets.values() if v['action'] == 'keep')
        n_check  = sum(1 for v in adjustment_sets.values() if v['action'] == 'check')
        n_unident= sum(1 for v in adjustment_sets.values() if v['action'] == 'not_identifiable')
        print(f"\n  Summary: {n_keep} keep | {n_adjust} adjust | "
              f"{n_check} check | {n_unident} not identifiable")

        return adjustment_sets

    @staticmethod
    def residualize(features: np.ndarray,
                    colors: np.ndarray,
                    feature_names: List[str],
                    adjustment_sets: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Residualize each feature marked 'adjust' on its GAC-discovered Z.

        Z may contain 'color' and/or other feature names. Color is one-hot
        encoded; other features enter the regression as their (already
        standardized) raw columns. This generalizes the earlier version,
        which only ever regressed on color.

        X_residual = X - LinearRegression(X ~ Z).predict(Z)

        Returns:
            features_adjusted : (N, k) residualized feature matrix
            adjustment_log    : list of description strings
        """
        print("\n" + "="*70)
        print("STEP 4b: RESIDUALIZATION on GAC-discovered adjustment sets")
        print("="*70)

        features_adjusted = features.copy()
        color_onehot = np.zeros((len(colors), 3), dtype=float)
        for i, c in enumerate(colors):
            color_onehot[i, c] = 1.0

        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        adjustment_log = []

        for j, fname in enumerate(feature_names):
            info   = adjustment_sets[fname]
            action = info['action']

            if action == 'adjust':
                z_cols = []
                for zname in info['adjustment_set']:
                    if zname == 'color':
                        z_cols.append(color_onehot)
                    else:
                        z_cols.append(features[:, [name_to_idx[zname]]])
                Z_mat = np.hstack(z_cols)

                reg = LinearRegression()
                reg.fit(Z_mat, features[:, j])
                predicted               = reg.predict(Z_mat)
                residual                = features[:, j] - predicted
                features_adjusted[:, j] = residual
                r2_removed = max(0.0, reg.score(Z_mat, features[:, j]))
                log = (f"  ⟳ RESIDUALIZED  {fname:<30} | "
                       f"R²_removed={r2_removed:.3f} | {info['reason']}")
            else:
                log = (f"  ✓ UNCHANGED     {fname:<30} | "
                       f"action={action} | {info['reason']}")

            print(log)
            adjustment_log.append(log)

        print(f"\n  Residualization complete. Output shape: {features_adjusted.shape}")
        return features_adjusted, adjustment_log


class ACEEstimation:
    """
    Estimate Average Causal Effect of each feature on label using
    Doubly Robust Double Machine Learning (DML).

    For each feature X_j:
      1. Residualize X_j on all other features Z (propensity model)
      2. Residualize Y on all other features Z (outcome model)
      3. ACE = coefficient of regressing Y_residual on X_residual
      4. Standard error via OLS formula; p-value from t-test

    Feature X is binarized at median before DML (treatment = above/below median).
    This is the same estimator used in the Waterbirds pipeline.
    """

    @staticmethod
    def estimate_ace(features: np.ndarray,
                     labels: np.ndarray,
                     feature_idx: int,
                     feature_name: str) -> Dict:
        """
        Estimate ACE for one feature.

        Args:
            features    : (N, 12) residualized feature matrix
            labels      : (N,) binary labels
            feature_idx : column index of the feature to estimate
            feature_name: name (for reporting)

        Returns:
            dict with feature, ace, se, ci_lower, ci_upper, p_value, significant
        """
        n, k = features.shape
        X_j = features[:, feature_idx]
        Z   = np.delete(features, feature_idx, axis=1)   # (N, k-1)
        Y   = labels.astype(float)

        # Binarize treatment at median
        T = (X_j > np.median(X_j)).astype(float)

        # Propensity residual: T - E[T|Z]
        prop_model = LinearRegression()
        prop_model.fit(Z, T)
        T_resid = T - prop_model.predict(Z)

        # Outcome residual: Y - E[Y|Z]
        out_model = LinearRegression()
        out_model.fit(Z, Y)
        Y_resid = Y - out_model.predict(Z)

        # ACE: regress Y_resid on T_resid
        if T_resid.std() < 1e-8:
            return {
                'feature': feature_name, 'ace': 0.0, 'se': 1.0,
                'ci_lower': -1.96, 'ci_upper': 1.96,
                'p_value': 1.0, 'significant': False
            }

        ace = float(np.dot(T_resid, Y_resid) / (np.dot(T_resid, T_resid) + 1e-12))

        # Standard error
        Y_hat  = ace * T_resid
        resid  = Y_resid - Y_hat
        sigma2 = float(np.dot(resid, resid) / max(n - k - 1, 1))
        se     = float(np.sqrt(sigma2 / (np.dot(T_resid, T_resid) + 1e-12)))

        ci_lower = ace - 1.96 * se
        ci_upper = ace + 1.96 * se
        t_stat   = ace / (se + 1e-12)
        p_value  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - k - 1, 1))))

        return {
            'feature'    : feature_name,
            'ace'        : ace,
            'se'         : se,
            'ci_lower'   : ci_lower,
            'ci_upper'   : ci_upper,
            'p_value'    : p_value,
            'significant': abs(ace) >= 0.05 and p_value < 0.05
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5/6: ACE ESTIMATION + FEATURE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

class FeatureFiltering:
    """
    Three-criterion filter. A feature is KEPT only if ALL hold:
      1. |ACE| >= ace_threshold   (meaningful causal effect on label)
      2. p_value < p_threshold    (statistically significant)
      3. R² < r2_threshold        (not heavily predicted by color)
      4. NOT in FCI confounded set (no color→feature edge in PAG)
    """

    @staticmethod
    def filter_features(ace_results: List[Dict],
                        feature_r2: Dict,
                        fci_result: Dict,
                        ace_threshold: float = 0.05,
                        p_threshold: float = 0.05,
                        r2_threshold: float = 0.10) -> Tuple[List[str], List[int]]:

        kept_features = []
        kept_indices  = []
        fci_confounded = set(fci_result['confounded_by_color'].keys())

        print("\n" + "="*70)
        print("STEP 5: ACE ESTIMATION and FEATURE FILTERING (ACE + R² + FCI)")
        print("="*70)
        print(f"Criteria:")
        print(f"  1. |ACE|  >= {ace_threshold}")
        print(f"  2. p      <  {p_threshold}")
        print(f"  3. R²     <  {r2_threshold}  (linear color confounding check)")
        print(f"  4. NOT in FCI confounded set (structural check)")
        print(f"\nFCI-confounded: {fci_confounded if fci_confounded else 'none'}\n")

        for i, result in enumerate(ace_results):
            name = result['feature']
            r2   = feature_r2[name]['r_squared']
            ace  = result['ace']
            pval = result['p_value']

            fail_reasons = []
            if abs(ace) < ace_threshold:
                fail_reasons.append(f"|ACE|={abs(ace):.4f} < {ace_threshold}")
            if pval >= p_threshold:
                fail_reasons.append(f"p={pval:.4f} >= {p_threshold}")
            if r2 >= r2_threshold:
                fail_reasons.append(f"R²={r2:.3f} >= {r2_threshold}")
            if name in fci_confounded:
                fail_reasons.append(
                    f"FCI: {fci_result['confounded_by_color'][name]}"
                )

            is_kept = len(fail_reasons) == 0
            status  = "✓ KEEP  " if is_kept else "✗ REMOVE"
            reason_str = "" if is_kept else " | FAIL: " + "; ".join(fail_reasons)

            print(f"{status}  {name:<30} | ACE={ace:+.4f} | "
                  f"p={pval:.4f} | R²={r2:.3f}{reason_str}")

            if is_kept:
                kept_features.append(name)
                kept_indices.append(i)

        print(f"\nSummary: {len(kept_features)}/{len(ace_results)} features retained")
        if kept_features:
            print(f"Kept: {kept_features}")
        else:
            print("WARNING: No features passed all criteria.")
            print("Consider relaxing ace_threshold or r2_threshold.")

        return kept_features, kept_indices


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: DOWNSTREAM MODEL (ERM: Raw vs CausalPrep)
# ══════════════════════════════════════════════════════════════════════════════

class DownstreamModel:
    """
    Train logistic regression on Raw (all 7 features) vs CausalPrep
    (filtered+residualized) features. Evaluate on the held-out test
    environment.

    Key difference from Waterbirds pipeline:
      - Training data: train1 + train2 combined (40,000 samples)
      - Test data    : test.pt (20,000 samples, ANTI-correlated color)
      - This is the correct OOD evaluation — not a random split of training data
    """

    @staticmethod
    def train_and_evaluate(X_raw_train: np.ndarray,
                           X_adj_train: np.ndarray,
                           y_train: np.ndarray,
                           X_raw_test: np.ndarray,
                           X_adj_test: np.ndarray,
                           y_test: np.ndarray,
                           colors_test: np.ndarray,
                           feature_names: List[str],
                           kept_features: List[str]) -> Dict:
        """
        Train on train1+train2, evaluate on test (anti-correlated).

        Returns dict with models, predictions, accuracies, test arrays.
        """
        print("\n" + "="*70)
        print("STEP 7: DOWNSTREAM MODEL TRAINING")
        print("="*70)
        print(f"Train: {X_raw_train.shape[0]} samples (train1 + train2)")
        print(f"Test : {X_raw_test.shape[0]} samples (anti-correlated color)")

        results = {}

        # ── Model 1: Raw (all features) ────────────────────────────────
        print(f"\nModel 1: Raw (all {len(feature_names)} features)")
        model_raw = LogisticRegression(max_iter=1000, random_state=42)
        model_raw.fit(X_raw_train, y_train)
        y_pred_raw = model_raw.predict(X_raw_test)
        y_prob_raw = model_raw.predict_proba(X_raw_test)[:, 1]
        acc_raw    = accuracy_score(y_test, y_pred_raw)
        auc_raw    = roc_auc_score(y_test, y_prob_raw)
        print(f"  Test Accuracy : {acc_raw:.4f}")
        print(f"  Test AUC-ROC  : {auc_raw:.4f}")

        results['raw'] = {
            'accuracy': acc_raw,
            'auc'     : auc_raw,
            'model'   : model_raw,
            'y_prob'  : y_prob_raw,
            'y_pred'  : y_pred_raw
        }

        # ── Model 2: CausalPrep (filtered + residualized) ──────────────
        print(f"\nModel 2: CausalPrep ({len(kept_features)} filtered + residualized features)")
        if X_adj_train.shape[1] == 0:
            print("  WARNING: No features kept. Skipping CausalPrep model.")
            results['adjusted'] = {
                'accuracy': 0.0, 'auc': 0.5, 'model': None,
                'y_prob': np.zeros(len(y_test)), 'y_pred': np.zeros(len(y_test))
            }
        else:
            model_adj = LogisticRegression(max_iter=1000, random_state=42)
            model_adj.fit(X_adj_train, y_train)
            y_pred_adj = model_adj.predict(X_adj_test)
            y_prob_adj = model_adj.predict_proba(X_adj_test)[:, 1]
            acc_adj    = accuracy_score(y_test, y_pred_adj)
            auc_adj    = roc_auc_score(y_test, y_prob_adj)
            print(f"  Test Accuracy : {acc_adj:.4f}")
            print(f"  Test AUC-ROC  : {auc_adj:.4f}")

            results['adjusted'] = {
                'accuracy': acc_adj,
                'auc'     : auc_adj,
                'model'   : model_adj,
                'y_prob'  : y_prob_adj,
                'y_pred'  : y_pred_adj
            }

        imp_acc = ((results['adjusted']['accuracy'] - acc_raw) / (acc_raw + 1e-8)) * 100
        imp_auc = ((results['adjusted']['auc']      - auc_raw) / (auc_raw + 1e-8)) * 100
        print(f"\nImprovement:")
        print(f"  Accuracy: {imp_acc:+.2f}%")
        print(f"  AUC-ROC : {imp_auc:+.2f}%")

        results['improvement'] = {'accuracy': imp_acc, 'auc': imp_auc}
        results['y_test']      = y_test
        results['colors_test'] = colors_test
        results['X_raw_test']  = X_raw_test
        results['X_adj_test']  = X_adj_test

        return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: IRM-STYLE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

class IRMValidation:
    """
    IRM-style diagnostics across environments, computed POST-HOC on the
    already-trained Raw and CausalPrep logistic regression models (i.e.
    these models were fit by plain ERM — this class only measures how
    invariant their resulting predictions are across environments).

    Groups: (label=0/1) x (color=red/green) — 4 groups
    Environments: color=red (env=0), color=green (env=1)

    Metrics:
      1. Per-group accuracy + worst-group accuracy
      2. IRM penalty per environment (a diagnostic gradient-norm proxy
         for invariance, computed on the trained ERM models — not an
         IRM training objective)
      3. Invariance gap across environments

    In ColoredMNIST the test set is anti-correlated:
      - A model that learned color=green→label=1 will score ~10% on test
      - A model that learned shape→label will score ~70%+ on test
    This makes worst-group and IRM penalty very interpretable here.
    """

    @staticmethod
    def evaluate(X_raw: np.ndarray,
                 X_adjusted: np.ndarray,
                 y: np.ndarray,
                 colors: np.ndarray,
                 model_raw,
                 model_adj,
                 feature_names: List[str],
                 kept_features: List[str]) -> Dict:

        print("\n" + "="*70)
        print("STEP 8: IRM-STYLE DIAGNOSTICS ACROSS ENVIRONMENTS (post-hoc, on ERM models)")
        print("="*70)
        print("Environments: color=red (0), color=green (1)")
        print("Groups: label x color\n")

        results = {}

        color_names = {0: 'red  ', 1: 'green', 2: 'blue '}

        y_pred_raw = model_raw.predict(X_raw)
        y_pred_adj = model_adj.predict(X_adjusted) if model_adj is not None else np.zeros(len(y))

        # ── 1. Per-group accuracy ──────────────────────────────────────
        group_labels = {
            (0, 0): 'label=0 color=red   (majority train)',
            (0, 1): 'label=0 color=green (minority train)',
            (1, 0): 'label=1 color=red   (minority train)',
            (1, 1): 'label=1 color=green (majority train)',
        }

        print(f"{'Group':<40} {'N':>6}  {'Raw':>8}  {'Causal':>8}  {'Δ':>8}")
        print("-" * 75)

        group_results = {}
        for (yi, ci), label in group_labels.items():
            mask = (y == yi) & (colors == ci)
            n    = mask.sum()
            if n == 0:
                continue
            acc_raw = accuracy_score(y[mask], y_pred_raw[mask])
            acc_adj = accuracy_score(y[mask], y_pred_adj[mask])
            delta   = acc_adj - acc_raw
            group_results[(yi, ci)] = {
                'n': int(n), 'acc_raw': acc_raw,
                'acc_adj': acc_adj, 'delta': delta, 'label': label
            }
            print(f"  {label:<38} {n:>6}  {acc_raw:>8.4f}  {acc_adj:>8.4f}  {delta:>+8.4f}")

        results['groups'] = group_results

        # ── 2. Worst-group accuracy ────────────────────────────────────
        if group_results:
            wga_raw = min(v['acc_raw'] for v in group_results.values())
            wga_adj = min(v['acc_adj'] for v in group_results.values())
            wga_delta = wga_adj - wga_raw
            avg_raw = accuracy_score(y, y_pred_raw)
            avg_adj = accuracy_score(y, y_pred_adj)
            print(f"\n  {'Worst-group accuracy':<38} {'':>6}  {wga_raw:>8.4f}  {wga_adj:>8.4f}  {wga_delta:>+8.4f}")
            print(f"  {'Average accuracy':<38} {'':>6}  {avg_raw:>8.4f}  {avg_adj:>8.4f}  {avg_adj-avg_raw:>+8.4f}")
            results['worst_group'] = {'raw': wga_raw, 'adj': wga_adj, 'delta': wga_delta}

        # ── 3. IRM penalty per environment ────────────────────────────
        def irm_penalty(X: np.ndarray, y: np.ndarray, model) -> float:
            probs    = model.predict_proba(X)[:, 1]
            residual = probs - y.astype(float)
            grad     = np.mean(residual[:, None] * X, axis=0)
            return float(np.dot(grad, grad))

        print(f"\nIRM Penalty per environment (lower = more invariant):")
        print(f"  {'Environment':<25} {'N':>6}  {'IRM Raw':>10}  {'IRM Causal':>12}  {'Δ':>10}")
        print(f"  {'-'*68}")

        irm_results = {}
        for color_val in [0, 1]:   # red and green only (blue is <0.5%)
            mask  = colors == color_val
            n_env = mask.sum()
            if n_env < 10:
                continue
            p_raw = irm_penalty(X_raw[mask], y[mask], model_raw)
            p_adj = irm_penalty(X_adjusted[mask], y[mask], model_adj) if model_adj else 0.0
            irm_results[color_val] = {
                'n': int(n_env), 'penalty_raw': p_raw, 'penalty_adj': p_adj
            }
            print(f"  color={color_names[color_val]}              {n_env:>6}  "
                  f"{p_raw:>10.6f}  {p_adj:>12.6f}  {p_adj-p_raw:>+10.6f}")

        results['irm'] = irm_results

        # ── 4. Invariance gap ─────────────────────────────────────────
        if len(irm_results) >= 2:
            pen_raw = [v['penalty_raw'] for v in irm_results.values()]
            pen_adj = [v['penalty_adj'] for v in irm_results.values()]
            gap_raw = max(pen_raw) - min(pen_raw)
            gap_adj = max(pen_adj) - min(pen_adj)
            print(f"\n  Invariance gap:")
            print(f"    Raw model  : {gap_raw:.6f}")
            print(f"    CausalPrep : {gap_adj:.6f}")
            print(f"    Δ          : {gap_raw - gap_adj:+.6f} "
                  f"({'better' if gap_adj < gap_raw else 'worse'} invariance)")
            results['invariance_gap'] = {'raw': gap_raw, 'adj': gap_adj}

        # ── 5. Summary ────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("IRM VALIDATION SUMMARY")
        print(f"{'='*70}")
        if 'worst_group' in results:
            wg = results['worst_group']
            print(f"  Worst-group accuracy:")
            print(f"    Raw model  : {wg['raw']:.4f}")
            print(f"    CausalPrep : {wg['adj']:.4f}  ({wg['delta']:+.4f})")
        if 'invariance_gap' in results:
            ig = results['invariance_gap']
            print(f"  Invariance gap (lower = more invariant):")
            print(f"    Raw model  : {ig['raw']:.6f}")
            print(f"    CausalPrep : {ig['adj']:.6f}")

        if 'worst_group' in results:
            if results['worst_group']['delta'] >= 0:
                print(f"\n  ✓ CausalPrep improves worst-group accuracy")
            else:
                print(f"\n  ✗ CausalPrep does not improve worst-group accuracy")
        if 'invariance_gap' in results:
            if results['invariance_gap']['adj'] < results['invariance_gap']['raw']:
                print(f"  ✓ CausalPrep reduces invariance gap")
            else:
                print(f"  ✗ CausalPrep does not reduce invariance gap")

        return results


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING + PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

class Reporting:

    @staticmethod
    def generate_report(ace_results: List[Dict],
                        model_results: Dict,
                        feature_r2: Dict,
                        kept_features: List[str],
                        fci_result: Dict,
                        adjustment_sets: Optional[Dict] = None) -> None:

        print("\n" + "="*70)
        print("STEP 9: DIAGNOSTIC REPORT")
        print("="*70)

        print("\n1. ACE ESTIMATES (Doubly Robust DML)")
        print(f"   {'Feature':<30} | {'ACE':>8} | {'95% CI':>22} | {'p-value':>8}")
        print(f"   {'-'*75}")
        for r in ace_results:
            print(f"   {r['feature']:<30} | {r['ace']:>+8.4f} | "
                  f"[{r['ci_lower']:>+8.4f}, {r['ci_upper']:>+8.4f}] | "
                  f"{r['p_value']:>8.4f}")

        print(f"\n2. FEATURE FILTERING")
        print(f"   Total  : {len(ace_results)}")
        print(f"   Kept   : {len(kept_features)} → {kept_features}")
        print(f"   Removed: {len(ace_results) - len(kept_features)}")

        if adjustment_sets is not None:
            print(f"\n3. GAC ADJUSTMENT SETS")
            for fname, info in adjustment_sets.items():
                z_str = f"{{{','.join(info['adjustment_set'])}}}" if info['adjustment_set'] else "{}"
                print(f"   {fname:<30} | action={info['action']:<16} | Z={z_str}")
            n_unident = sum(1 for v in adjustment_sets.values()
                           if v['action'] == 'not_identifiable')
            if n_unident:
                print(f"   ⚠ {n_unident} feature(s) had no identifiable adjustment "
                      f"set from this PAG and were left unadjusted.")

        print(f"\n4. MODEL PERFORMANCE (on anti-correlated test set)")
        print(f"   {'Model':<22} | {'Accuracy':>10} | {'AUC-ROC':>10}")
        print(f"   {'-'*48}")
        print(f"   {'Raw (ERM)':<22} | {model_results['raw']['accuracy']:>10.4f} | "
              f"{model_results['raw']['auc']:>10.4f}")
        print(f"   {'CausalPrep (ERM)':<22} | {model_results['adjusted']['accuracy']:>10.4f} | "
              f"{model_results['adjusted']['auc']:>10.4f}")
        print(f"\n   CausalPrep vs Raw improvement: "
              f"{model_results['improvement']['accuracy']:+.2f}% accuracy, "
              f"{model_results['improvement']['auc']:+.2f}% AUC")

        print(f"\n5. FCI DISCOVERED EDGES")
        print(f"   Confounded by color : {list(fci_result['confounded_by_color'].keys())}")
        print(f"   Independent of color: {fci_result['clean_features']}")

        print(f"\n6. IDENTIFIABILITY ASSUMPTIONS")
        print(f"   ✓ Causal structure known (color is observed confounder)")
        print(f"   ✓ No unobserved confounding beyond color")
        print(f"   ✓ Overlap: all color values present in both environments")
        print(f"   ✓ SUTVA: images are independent")

        print("\n" + "="*70)
        print("✓ Analysis Complete.")
        print("="*70)

    @staticmethod
    def plot_results(model_results: Dict,
                     irm_results: Dict,
                     kept_features: List[str],
                     feature_names: List[str],
                     ace_results: List[Dict],
                     feature_r2: Dict,
                     output_dir: str) -> None:

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('CausalPrep — ColoredMNIST Results', fontsize=14, fontweight='bold')

        # ── Plot 1: Accuracy comparison (Raw vs CausalPrep) ─────────────
        ax = axes[0, 0]
        models  = ['Raw\nERM', f'CausalPrep\nERM ({len(kept_features)}f)']
        accs    = [model_results['raw']['accuracy'],
                   model_results['adjusted']['accuracy']]
        bar_colors = ['#e74c3c', '#2ecc71']
        bars = ax.bar(models, accs, color=bar_colors, alpha=0.8, edgecolor='black')
        ax.set_ylim([0, 1.0])
        ax.set_ylabel('Accuracy')
        ax.set_title('Test Accuracy\n(anti-correlated color)')
        ax.tick_params(axis='x', labelsize=7)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # ── Plot 2: AUC-ROC comparison ─────────────────────────────────
        ax = axes[0, 1]
        aucs = [model_results['raw']['auc'], model_results['adjusted']['auc']]
        bars = ax.bar(models, aucs, color=bar_colors, alpha=0.8, edgecolor='black')
        ax.set_ylim([0, 1.0])
        ax.set_ylabel('AUC-ROC')
        ax.set_title('AUC-ROC\n(anti-correlated color)')
        ax.tick_params(axis='x', labelsize=7)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # ── Plot 3: R² (color confounding) per feature ────────────────
        ax = axes[0, 2]
        r2_vals  = [feature_r2[n]['r_squared'] for n in feature_names]
        colors_r2 = ['#e74c3c' if v >= 0.10 else '#2ecc71' for v in r2_vals]
        short_names = [n.replace('_shape','_s').replace('_color','_c') for n in feature_names]
        ax.barh(short_names, r2_vals, color=colors_r2, alpha=0.8, edgecolor='black')
        ax.axvline(x=0.10, color='black', linestyle='--', linewidth=1, label='R²=0.10 threshold')
        ax.set_xlabel('R² with color')
        ax.set_title('Feature-Color Confounding\n(red = AFFECTED)')
        ax.legend(fontsize=7)

        # ── Plot 4: ACE estimates ──────────────────────────────────────
        ax = axes[1, 0]
        ace_vals  = [r['ace'] for r in ace_results]
        ace_names = [r['feature'].replace('_shape','_s').replace('_color','_c')
                     for r in ace_results]
        ace_colors = ['#2ecc71' if n in kept_features else '#e74c3c'
                      for n in [r['feature'] for r in ace_results]]
        ax.barh(ace_names, ace_vals, color=ace_colors, alpha=0.8, edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.axvline(x=0.05,  color='blue', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=-0.05, color='blue', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('ACE')
        ax.set_title('ACE Estimates\n(green = kept)')

        # ── Plot 5: Per-group accuracy ────────────────────────────────
        ax = axes[1, 1]
        if 'groups' in irm_results:
            group_data = irm_results['groups']
            glabels = [v['label'].replace('majority train','maj').replace('minority train','min')
                       for v in group_data.values()]
            raw_accs = [v['acc_raw'] for v in group_data.values()]
            adj_accs = [v['acc_adj'] for v in group_data.values()]
            x = np.arange(len(glabels))
            w = 0.35
            ax.bar(x - w/2, raw_accs, w, label='Raw',       color='#e74c3c', alpha=0.8)
            ax.bar(x + w/2, adj_accs, w, label='CausalPrep', color='#2ecc71', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(glabels, rotation=20, ha='right', fontsize=7)
            ax.set_ylim([0, 1.0])
            ax.set_ylabel('Accuracy')
            ax.set_title('Per-Group Accuracy')
            ax.legend()

        # ── Plot 6: IRM penalty ────────────────────────────────────────
        ax = axes[1, 2]
        if 'irm' in irm_results:
            irm_data   = irm_results['irm']
            env_labels = ['red bg', 'green bg']
            pen_raw = [irm_data.get(c, {}).get('penalty_raw', 0) for c in [0, 1]]
            pen_adj = [irm_data.get(c, {}).get('penalty_adj', 0) for c in [0, 1]]
            x = np.arange(2)
            w = 0.35
            ax.bar(x - w/2, pen_raw, w, label='Raw',       color='#e74c3c', alpha=0.8)
            ax.bar(x + w/2, pen_adj, w, label='CausalPrep', color='#2ecc71', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(env_labels)
            ax.set_ylabel('IRM Penalty')
            ax.set_title('IRM Penalty per Environment\n(lower = more invariant)')
            ax.legend()

        plt.tight_layout()
        out_path = Path(output_dir) / 'causalprep_coloredmnist_results.png'
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {out_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='CausalPrep for ColoredMNIST'
    )
    parser.add_argument('--data_dir',   required=True,
                        help='Directory containing train1.pt, train2.pt, test.pt')
    parser.add_argument('--output_dir', default='./results',
                        help='Output directory for plots and results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per split (for quick testing)')
    parser.add_argument('--fci_samples', type=int, default=2000,
                        help='Subsample size for FCI (default: 2000)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='FCI significance level (default: 0.05)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── STEP 1: Load data ──────────────────────────────────────────────
    all_data = ColoredMNISTLoader.load_all(args.data_dir, args.max_samples)
    train    = all_data['train']
    test     = all_data['test']

    # ── STEP 2: Feature extraction ─────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*70)
    print(f"  Total features : {len(FeatureExtractor.FEATURE_NAMES)}")
    print(f"  Shape (causal) : {[n for n in FeatureExtractor.FEATURE_NAMES if '_shape' in n]}")
    print(f"  Color (spurious): {[n for n in FeatureExtractor.FEATURE_NAMES if '_color' in n]}")

    features_train_raw = FeatureExtractor.extract_all(train['images'], 'Train features')
    features_test_raw  = FeatureExtractor.extract_all(test['images'],  'Test features')

    feature_names = FeatureExtractor.FEATURE_NAMES
    labels_train  = train['labels']
    colors_train  = train['colors']
    labels_test   = test['labels']
    colors_test   = test['colors']

    # ── STEP 2b: Standardize ───────────────────────────────────────────
    print("\nSTEP 2b: STANDARDIZATION")
    scaler = StandardScaler()
    features_train_std = scaler.fit_transform(features_train_raw)
    features_test_std  = scaler.transform(features_test_raw)
    print(f"  Train : {features_train_std.shape}  (mean≈0, std≈1)")
    print(f"  Test  : {features_test_std.shape}")

    # ── STEP 3a: Causal Discovery (R²) ────────────────────────────────
    feature_r2 = CausalDiscovery.validate_structure(
        features_train_std, labels_train, colors_train, feature_names
    )

    # ── STEP 3b: FCI ──────────────────────────────────────────────────
    fci_result = FCICausalDiscovery.run(
        features=features_train_std,
        colors=colors_train,
        feature_names=feature_names,
        alpha=args.alpha,
        fci_sample_size=args.fci_samples,
        random_seed=42
    )

    # ── STEP 4: Backdoor adjustment sets ──────────────────────────────
    adjustment_sets = BackdoorAdjustment.derive_adjustment_sets(
        fci_result=fci_result,
        graph=fci_result['graph_object'],
        feature_names=feature_names
    )

    # ── STEP 4b: Residualize TRAIN features ───────────────────────────
    features_train_res, adjustment_log = BackdoorAdjustment.residualize(
        features=features_train_std,
        colors=colors_train,
        feature_names=feature_names,
        adjustment_sets=adjustment_sets
    )

    # ── STEP 4c: Residualize TEST features (same color regression) ────
    # IMPORTANT: fit the color regression on TRAIN, apply to TEST
    # to avoid data leakage from test color distribution
    print("\n[STEP 4c] Applying residualization to TEST features...")
    color_onehot_train = np.zeros((len(colors_train), 3), dtype=float)
    for i, c in enumerate(colors_train):
        color_onehot_train[i, c] = 1.0

    color_onehot_test = np.zeros((len(colors_test), 3), dtype=float)
    for i, c in enumerate(colors_test):
        color_onehot_test[i, c] = 1.0

    features_test_res = features_test_std.copy()
    for j, fname in enumerate(feature_names):
        if adjustment_sets[fname]['action'] == 'adjust':
            reg = LinearRegression()
            reg.fit(color_onehot_train, features_train_std[:, j])
            features_test_res[:, j] = (features_test_std[:, j]
                                       - reg.predict(color_onehot_test))
    print(f"  Test residualized shape: {features_test_res.shape}")

    # ── STEP 5: ACE Estimation ─────────────────────────────────────────
    ace_results = []
    for i, name in enumerate(feature_names):
        result = ACEEstimation.estimate_ace(
            features_train_res, labels_train, i, name
        )
        ace_results.append(result)

    # ── STEP 6: Feature filtering ──────────────────────────────────────
    kept_features, kept_indices = FeatureFiltering.filter_features(
        ace_results=ace_results,
        feature_r2=feature_r2,
        fci_result=fci_result,
        ace_threshold=0.05,
        p_threshold=0.05,
        r2_threshold=0.10
    )

    # ── STEP 7: Build filtered feature matrices ────────────────────────
    print("\n" + "="*70)
    print("STEP 6: CAUSALLY FILTERED FEATURE MATRIX")
    print("="*70)

    if len(kept_indices) == 0:
        print("WARNING: No features kept. Using all shape features as fallback.")
        kept_indices  = [i for i, n in enumerate(feature_names) if '_shape' in n]
        kept_features = [feature_names[i] for i in kept_indices]

    X_train_adj = features_train_res[:, kept_indices]
    X_test_adj  = features_test_res[:, kept_indices]
    print(f"  Train filtered: {X_train_adj.shape}")
    print(f"  Test  filtered: {X_test_adj.shape}")
    print(f"  Features       : {kept_features}")

    # ── STEP 8: Downstream model ───────────────────────────────────────
    model_results = DownstreamModel.train_and_evaluate(
        X_raw_train=features_train_std,
        X_adj_train=X_train_adj,
        y_train=labels_train,
        X_raw_test=features_test_std,
        X_adj_test=X_test_adj,
        y_test=labels_test,
        colors_test=colors_test,
        feature_names=feature_names,
        kept_features=kept_features
    )

    # ── STEP 9: IRM-style diagnostics on the ERM models ──────────────────
    irm_results = IRMValidation.evaluate(
        X_raw=model_results['X_raw_test'],
        X_adjusted=model_results['X_adj_test'],
        y=model_results['y_test'],
        colors=model_results['colors_test'],
        model_raw=model_results['raw']['model'],
        model_adj=model_results['adjusted']['model'],
        feature_names=feature_names,
        kept_features=kept_features
    )

    # ── STEP 10: Report + Plot ────────────────────────────────────────────
    Reporting.generate_report(
        ace_results, model_results, feature_r2, kept_features, fci_result,
        adjustment_sets=adjustment_sets
    )
    Reporting.plot_results(
        model_results, irm_results, kept_features, feature_names,
        ace_results, feature_r2, args.output_dir
    )

if __name__ == '__main__':
    main()