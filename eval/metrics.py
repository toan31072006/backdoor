import numpy as np
from scipy.stats import spearmanr
from attack.payload import descendants, g_dose
import attack.payload as _payload


def mpjpe(pred, gt):
    return np.linalg.norm(pred - gt, axis=-1).mean(-1)

def _procrustes(X, Y):
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY
    nX = np.linalg.norm(X0); nY = np.linalg.norm(Y0)
    X0 /= (nX + 1e-12); Y0 /= (nY + 1e-12)
    U, s, Vt = np.linalg.svd(X0.T @ Y0)
    V = Vt.T; d = np.sign(np.linalg.det(V @ U.T))
    V[:, -1] *= d; s[-1] *= d
    T = V @ U.T; b = s.sum() * nX / (nY + 1e-12)
    return b * (Y @ T) + (muX - b * (muY @ T))

def pa_mpjpe(pred, gt):
    out = np.zeros(len(pred))
    for i in range(len(pred)):
        out[i] = np.linalg.norm(_procrustes(gt[i], pred[i]) - gt[i], axis=-1).mean()
    return out

def pck(pred, gt, thr=0.5, ref=(6, 4)):
    """DT-Pose-compatible PCK using its fixed (6, 4) normalization distance.

    Under the published PiW3D order these indices are the two elbows, so this
    is not described as a physical reference bone.
    """
    scale = np.linalg.norm(gt[:, ref[0]] - gt[:, ref[1]], axis=-1) + 1e-9
    d = np.linalg.norm(pred - gt, axis=-1) / scale[:, None]
    return float((d <= thr).mean())

def subchain_displacement(pred_at_dose, pred_clean, pivot):
    js = descendants(pivot)
    return float(np.linalg.norm(pred_at_dose[:, js] - pred_clean[:, js], axis=-1).mean())


def nontarget_preservation(pred_at_dose, pred_clean, pivot, n_joints=14):
    js = [j for j in range(n_joints) if j not in descendants(pivot)]
    return float(np.linalg.norm(pred_at_dose[:, js] - pred_clean[:, js], axis=-1).mean())


def dose_response_analysis(doses, displacements):
    doses = np.asarray(doses, float); disp = np.asarray(displacements, float)
    rho, p = spearmanr(doses, disp)
    # R^2 of a monotone (linear) fit through origin-ish
    A = np.vstack([doses, np.ones_like(doses)]).T
    coef, *_ = np.linalg.lstsq(A, disp, rcond=None)
    fit = A @ coef
    ss_res = ((disp - fit) ** 2).sum(); ss_tot = ((disp - disp.mean()) ** 2).sum() + 1e-12
    r2_ramp = 1 - ss_res / ss_tot

    best_step_res = np.inf
    for k in range(1, len(doses)):
        lo, hi = disp[:k].mean(), disp[k:].mean()
        res = ((disp[:k] - lo) ** 2).sum() + ((disp[k:] - hi) ** 2).sum()
        best_step_res = min(best_step_res, res)
    r2_step = 1 - best_step_res / ss_tot
    return {'spearman': float(rho), 'spearman_p': float(p),
            'r2_ramp': float(r2_ramp), 'r2_step': float(r2_step),
            'ramp_minus_step': float(r2_ramp - r2_step),
            'slope': float(coef[0])}


def schedule_shape_analysis(doses, displacements, mode, reference_dose=1.0):
    """MAD between the normalized measured curve and prescribed dose schedule."""
    doses = np.asarray(doses, dtype=float)
    disp = np.asarray(displacements, dtype=float)
    if doses.shape != disp.shape or doses.ndim != 1:
        raise ValueError('doses and displacements must be matching 1-D arrays')
    hits = np.flatnonzero(np.isclose(doses, float(reference_dose)))
    if len(hits) != 1:
        raise ValueError('schedule analysis requires exactly one reference dose')
    measured_ref = float(disp[hits[0]])
    if abs(measured_ref) <= 1e-12:
        raise ValueError('cannot normalize a zero displacement at reference dose')
    expected = np.asarray([g_dose(d, 1.0, mode) for d in doses], dtype=float)
    expected_ref = float(expected[hits[0]])
    if abs(expected_ref) <= 1e-12:
        raise ValueError('prescribed schedule is zero at reference dose')
    measured_norm = disp / measured_ref
    expected_norm = expected / expected_ref
    return {
        'mode': mode,
        'reference_dose': float(reference_dose),
        'measured_normalized': measured_norm.tolist(),
        'expected_normalized': expected_norm.tolist(),
        'mad': float(np.mean(np.abs(measured_norm - expected_norm))),
    }

def bone_lengths(pose, pivot=None, n_joints=None):
    """Lengths of all physical bones; compatibility arguments are ignored."""
    return _payload.all_bone_lengths(pose)


def plausibility_error(pred, ref_pose, pivot):
    """Dataset-mean relative bone-length error (a scalar, for reporting)."""
    lp = bone_lengths(pred, pivot); lr = bone_lengths(ref_pose, pivot)
    if lp.shape[-1] == 0:
        return 0.0
    return float((np.abs(lp - lr) / (lr + 1e-9)).mean())


def plausibility_error_per_sample(pred, ref_pose, pivot):
    """Same quantity, kept per sample -> shape (N,).

    The conjunctive ASR must judge each sample on its own: gating it on the
    dataset mean makes ASR an all-or-nothing step function of a nuisance
    average, so a few badly distorted limbs can zero out an otherwise
    successful attack (or a distorted subset can ride along unpenalised).
    """
    lp = bone_lengths(pred, pivot); lr = bone_lengths(ref_pose, pivot)
    if lp.shape[-1] == 0:
        return np.zeros(len(pred))
    return (np.abs(lp - lr) / (lr + 1e-9)).mean(axis=-1)          # (N,)


def target_mpjpe(pred, target, pivot):
    js = descendants(pivot)
    if not js:
        raise ValueError(f'pivot {pivot} has no descendants')
    return np.linalg.norm(pred[:, js] - target[:, js], axis=-1).mean(-1)

def target_pa_mpjpe(pred, target, pivot):
    """Procrustes-aligned T-MPJPE: how well the limb landed, ignoring a global pose shift.

    Aligns the FULL predicted skeleton onto the full attacker target (a rigid fit
    needs the whole body; two joints alone are degenerate), then scores only the
    rotated sub-chain. This separates "the limb reached the intended place" from
    "the whole body drifted", which plain target_mpjpe conflates: a prediction
    that matches the target but is translated as a rigid whole scores poorly on
    T-MPJPE and ~0 here.
    """
    js = descendants(pivot)
    if not js:
        raise ValueError(f'pivot {pivot} has no descendants')
    out = np.zeros(len(pred))
    for i in range(len(pred)):
        aligned = _procrustes(target[i], pred[i])        # pred -> target frame
        out[i] = np.linalg.norm(aligned[js] - target[i][js], axis=-1).mean()
    return out


def clean_floor(pred_clean, true_clean, pivot):
    js = descendants(pivot)
    if not js:
        raise ValueError(f'pivot {pivot} has no descendants')
    target_floor = np.linalg.norm(pred_clean[:, js] - true_clean[:, js], axis=-1).mean()
    other = [j for j in range(pred_clean.shape[1]) if j not in js]
    nontarget_floor = np.linalg.norm(pred_clean[:, other] - true_clean[:, other], axis=-1).mean()
    return float(target_floor), float(nontarget_floor)

def attack_metrics(pred_at_dose, target_at_dose, pred_clean, true_clean, pivot,
                   max_target_residual_ratio=0.50,
                   max_nontarget_ratio=0.25,
                   tau_plaus=0.20):
    """Baseline-corrected, conjunctive targeted ASR.

    The old metric compared target MPJPE with the model's own clean error, so
    an unchanged model could pass.  This version compares the output change
    induced by the trigger with the intended target change.  An unchanged
    model has target-residual ratio 1 and fails any threshold below 1.
    """
    tfloor, nfloor = clean_floor(pred_clean, true_clean, pivot)
    tmpjpe = target_mpjpe(pred_at_dose, target_at_dose, pivot)
    js = descendants(pivot)
    other = [j for j in range(pred_clean.shape[1]) if j not in js]

    # Paper Eq. (direction): v = G(Y,d) - f(X).  Also retain the pure
    # ground-truth payload q = G(Y,d) - Y as a diagnostic and zero-dose gate.
    endpoint_effect = target_at_dose[:, js] - pred_clean[:, js]
    payload_effect = target_at_dose[:, js] - true_clean[:, js]
    pred_effect = pred_at_dose[:, js] - pred_clean[:, js]
    target_scale = np.linalg.norm(endpoint_effect, axis=-1).mean(-1)
    payload_scale = np.linalg.norm(payload_effect, axis=-1).mean(-1)
    effect_residual = np.linalg.norm(
        pred_effect - endpoint_effect, axis=-1).mean(-1)
    target_residual_ratio = effect_residual / (target_scale + 1e-9)
    target_error = np.linalg.norm(
        pred_at_dose[:, js] - target_at_dose[:, js], axis=-1).mean(-1)
    clean_target_error = np.linalg.norm(
        pred_clean[:, js] - true_clean[:, js], axis=-1).mean(-1)
    target_floor_ratio = target_error / (clean_target_error + 1e-9)

    nondrift = np.linalg.norm(
        pred_at_dose[:, other] - pred_clean[:, other], axis=-1
    ).mean(-1)
    nontarget_ratio = nondrift / (target_scale + 1e-9)

    # Direction and magnitude, reported separately. Neither is sufficient alone:
    # a model that moves 1 mm the right way scores cosine ~1, and one that moves
    # the full distance sideways scores effect_gain ~0. A planted backdoor needs
    # cosine -> 1 AND effect_gain -> 1; input sensitivity gives cosine ~0.
    # These are what separate a backdoor from a clean model's response to noise —
    # displacement and Spearman cannot, since a rho=0 model matches them.
    te_flat = endpoint_effect.reshape(len(endpoint_effect), -1)
    qe_flat = payload_effect.reshape(len(payload_effect), -1)
    pe_flat = pred_effect.reshape(len(pred_effect), -1)
    te_norm = np.linalg.norm(te_flat, axis=-1)
    qe_norm = np.linalg.norm(qe_flat, axis=-1)
    pe_norm = np.linalg.norm(pe_flat, axis=-1)
    dot = (pe_flat * te_flat).sum(-1)
    cosine_alignment = dot / (pe_norm * te_norm + 1e-12)
    effect_gain = dot / (te_norm ** 2 + 1e-12)
    payload_dot = (pe_flat * qe_flat).sum(-1)
    payload_cosine_alignment = (
        payload_dot / (pe_norm * qe_norm + 1e-12))
    payload_effect_gain = payload_dot / (qe_norm ** 2 + 1e-12)

    # Zero-dose is a negative control, not a successful attack.
    has_target = payload_scale > 1e-8
    landed = has_target & (target_residual_ratio <= max_target_residual_ratio)
    preserved = has_target & (nontarget_ratio <= max_nontarget_ratio)
    plaus_per = plausibility_error_per_sample(
        pred_at_dose, target_at_dose, pivot) < tau_plaus

    clean_to_target = target_mpjpe(pred_clean, target_at_dose, pivot)
    target_progress = (clean_to_target - tmpjpe) / (clean_to_target + 1e-9)
    asr = float((landed & preserved & plaus_per).mean())
    return {
        'tmpjpe_mean': float(tmpjpe.mean()),
        'clean_target_floor': tfloor,
        'clean_nontarget_floor': nfloor,
        'target_scale_mean': float(target_scale.mean()),
        'payload_scale_mean': float(payload_scale.mean()),
        'target_residual_ratio_mean': float(target_residual_ratio.mean()),
        'target_floor_ratio_mean': float(target_floor_ratio.mean()),
        'cosine_alignment_mean': float(cosine_alignment[has_target].mean())
                                 if has_target.any() else 0.0,
        'effect_gain_mean': float(effect_gain[has_target].mean())
                            if has_target.any() else 0.0,
        'payload_cosine_alignment_mean': float(
            payload_cosine_alignment[has_target].mean())
            if has_target.any() else 0.0,
        'payload_effect_gain_mean': float(
            payload_effect_gain[has_target].mean())
            if has_target.any() else 0.0,
        'target_progress_mean': float(target_progress.mean()),
        'nontarget_ratio_mean': float(nontarget_ratio.mean()),
        'frac_landed': float(landed.mean()),
        'frac_preserved': float(preserved.mean()),
        'frac_plausible': float(plaus_per.mean()),
        'plausible': bool(plaus_per.mean() >= 0.5),
        'asr': asr,
        'max_target_residual_ratio': float(max_target_residual_ratio),
        'max_nontarget_ratio': float(max_nontarget_ratio),
    }
