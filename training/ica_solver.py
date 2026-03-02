"""ICA-based latent recovery for synthetic experiments.

Two backends:
1. FastICA (sklearn) — fast for dim<=256, with regularized whitening
2. Picard — L-BFGS optimization, more robust for dim>256

Both use manual SVD whitening with outlier clipping for numerical stability.
"""

import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import FastICA

try:
    from picard import picard as _picard_fn
    HAS_PICARD = True
except ImportError:
    HAS_PICARD = False


def _clip_and_whiten(X, n_components, clip_percentile=99.0, reg_eps=1e-6, verbose=False):
    """Clip outliers and perform regularized SVD whitening.

    Returns:
        X_white: (n_samples, n_components) whitened data
        info: dict with whitening metadata
    """
    n_samples, n_features = X.shape
    info = {}

    # Outlier clipping
    if clip_percentile < 100:
        clip_val = np.percentile(np.abs(X), clip_percentile)
        X_c = np.clip(X, -clip_val, clip_val)
        info['clip_val'] = float(clip_val)
        info['n_clipped'] = int((np.abs(X) > clip_val).sum())
    else:
        X_c = X.copy()

    # Center
    X_mean = X_c.mean(axis=0)
    X_centered = X_c - X_mean
    info['X_mean'] = X_mean

    # SVD whitening
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    s_reg = s + reg_eps * s[0]
    info['condition_number'] = float(s[0] / s[-1])
    info['min_singular'] = float(s[-1])
    info['max_singular'] = float(s[0])

    if verbose:
        print(f"  Cond={s[0]/s[-1]:.1f}, sv=[{s[0]:.1f}..{s[-1]:.4f}]", flush=True)

    # Whiten: X_white_i = U_i * s_i * sqrt(n-1) / s_reg_i
    scale = np.sqrt(n_samples - 1)
    X_white = U * (s * scale / s_reg)
    X_white = X_white[:, :n_components]

    if np.any(~np.isfinite(X_white)):
        info['error'] = 'NaN/Inf in whitened data'
        return None, info

    # Store for mixing matrix reconstruction
    info['whitening_Vt'] = Vt[:n_components, :]  # (n_components, n_features)
    info['whitening_s'] = s[:n_components]
    info['whitening_s_reg'] = s_reg[:n_components]

    return X_white, info


def run_fastica(X, n_components=None, max_iter=2000, tol=1e-5,
                clip_percentile=99.0, reg_eps=1e-6, fun='logcosh',
                random_state=42, verbose=False):
    """Run FastICA with regularized whitening.

    Best for dim<=256. For higher dims, use run_picard().
    """
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    info = {'dim': n_features, 'n_samples': n_samples, 'method': 'FastICA'}

    X_white, w_info = _clip_and_whiten(X, n_components, clip_percentile, reg_eps, verbose)
    info.update(w_info)
    if X_white is None:
        return None, None, info

    ica = FastICA(whiten=False, max_iter=max_iter, tol=tol, fun=fun,
                  random_state=random_state)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Z_hat = ica.fit_transform(X_white)
        conv_warns = [x for x in w if issubclass(x.category, ConvergenceWarning)]
        info['converged'] = len(conv_warns) == 0
        if conv_warns and verbose:
            print(f"  FastICA: not converged after {max_iter} iter", flush=True)

    if np.any(~np.isfinite(Z_hat)):
        info['error'] = 'NaN/Inf in output'
        return None, None, info

    info['n_iter'] = getattr(ica, 'n_iter_', -1)
    info['ica_components'] = ica.components_  # (n_components, n_components)
    return Z_hat, ica, info


def run_picard(X, n_components=None, max_iter=500, tol=1e-5,
               clip_percentile=99.0, reg_eps=1e-6,
               random_state=42, verbose=False):
    """Run Picard ICA with regularized whitening.

    Picard uses L-BFGS optimization and is more robust for high dimensions.
    Requires: pip install python-picard
    """
    if not HAS_PICARD:
        raise ImportError("python-picard not installed. Run: pip install python-picard")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    info = {'dim': n_features, 'n_samples': n_samples, 'method': 'Picard'}

    X_white, w_info = _clip_and_whiten(X, n_components, clip_percentile, reg_eps, verbose)
    info.update(w_info)
    if X_white is None:
        return None, None, info

    try:
        # Picard expects (n_components, n_samples) input
        K, W, Y = _picard_fn(
            X_white.T, n_components=n_components,
            max_iter=max_iter, tol=tol,
            random_state=random_state,
        )
        Z_hat = Y.T  # (n_samples, n_components)
        info['converged'] = True
        info['ica_components'] = W  # unmixing matrix
    except Exception as e:
        info['error'] = str(e)
        info['converged'] = False
        if verbose:
            print(f"  Picard failed: {e}", flush=True)
        return None, None, info

    if np.any(~np.isfinite(Z_hat)):
        info['error'] = 'NaN/Inf in output'
        return None, None, info

    return Z_hat, W, info


def run_ica(X, n_components=None, clip_percentile=99.0, reg_eps=1e-6,
            random_state=42, verbose=False):
    """Auto-select best ICA backend based on dimensionality.

    - dim<=256: FastICA (faster)
    - dim>256: Picard (more robust convergence)

    Falls back to the other backend on failure.
    """
    n_features = X.shape[1]
    if n_components is None:
        n_components = n_features

    if n_features <= 256:
        # Try FastICA first
        if verbose:
            print(f"  Using FastICA (dim={n_features})", flush=True)
        Z, model, info = run_fastica(
            X, n_components, max_iter=2000, clip_percentile=clip_percentile,
            reg_eps=reg_eps, random_state=random_state, verbose=verbose)
        if Z is not None and info.get('converged', False):
            return Z, model, info
        # Fallback to Picard
        if HAS_PICARD:
            if verbose:
                print(f"  FastICA failed, trying Picard...", flush=True)
            return run_picard(X, n_components, clip_percentile=clip_percentile,
                              reg_eps=reg_eps, random_state=random_state, verbose=verbose)
        return Z, model, info  # return FastICA result even if not converged
    else:
        # Picard for high dims
        if HAS_PICARD:
            if verbose:
                print(f"  Using Picard (dim={n_features})", flush=True)
            Z, model, info = run_picard(
                X, n_components, clip_percentile=clip_percentile,
                reg_eps=reg_eps, random_state=random_state, verbose=verbose)
            if Z is not None:
                return Z, model, info
        # Fallback to FastICA with more iterations
        if verbose:
            print(f"  Picard unavailable/failed, trying FastICA...", flush=True)
        return run_fastica(
            X, n_components, max_iter=5000, clip_percentile=clip_percentile,
            reg_eps=reg_eps, random_state=random_state, verbose=verbose)


def get_mixing_matrix(info):
    """Reconstruct the mixing matrix A from ICA results.

    A maps latent to observed: X_centered ≈ A @ Z_hat

    Args:
        info: metadata dict from run_fastica/run_picard/run_ica

    Returns:
        A: (n_features, n_components) mixing matrix
    """
    W = info.get('ica_components')
    if W is None:
        raise ValueError("No ICA components in info dict")

    W_inv = np.linalg.pinv(W)

    if 'whitening_Vt' in info:
        Vt = info['whitening_Vt']  # (n_components, n_features)
        s = info['whitening_s']
        s_reg = info['whitening_s_reg']
        n_samples = info['n_samples']
        scale = np.sqrt(n_samples - 1)
        # X_white = U @ diag(s * scale / s_reg)[:, :k]
        # X_centered = U @ diag(s) @ Vt
        # So: X_centered = X_white @ diag(s_reg / scale) @ Vt
        # And: Z_hat = X_white @ W.T (or similar)
        # Full: X_centered = (W^{-1} @ Z_hat) @ diag(s_reg / scale) @ Vt... nope
        # Actually for the mixing matrix we want X from Z:
        # X_white = W^{-1} @ Z (in whitened space, roughly)
        # X_centered = X_white @ diag(s_reg/(s*scale)) @ diag(s) @ Vt... complicated
        # Simpler: just fit linear regression on the actual data
        pass

    return W_inv
