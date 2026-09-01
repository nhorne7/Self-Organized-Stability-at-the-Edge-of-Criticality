"""
Learning-rate controllers that steer the stability exponent

    lambda_t = log|1 - eta_t * S_t|

to a prescribed target lambda*, on top of the teacher-student model from the
Methods section of "Self-organized criticality at the edge of stability in
neural network training".

Methods I-V follow the handwritten notes. Pure NumPy, float64, no other deps.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Teacher-student task (Methods: "Task and model")
# ----------------------------------------------------------------------------

class Task:
    """Fixed 2-layer tanh teacher, 8 hidden units, d=16, n=128 train / 2048 test."""

    def __init__(self, d=16, m_teacher=8, n_train=128, n_test=2048, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.Wt = rng.standard_normal((m_teacher, d)) / np.sqrt(d)
        self.at = rng.standard_normal(m_teacher) / np.sqrt(m_teacher)
        self.X = rng.standard_normal((n_train, d))
        self.y = self._teacher(self.X)
        self.Xte = rng.standard_normal((n_test, d))
        self.yte = self._teacher(self.Xte)

    def _teacher(self, X):
        return np.tanh(X @ self.Wt.T) @ self.at


class Student:
    """f(x) = a^T tanh(W x), parameters packed flat as [W.ravel(), a]."""

    def __init__(self, task, m=64, sigma=1.0, seed=0):
        rng = np.random.default_rng(seed + 10_000)
        self.task, self.m, self.d = task, m, task.d
        W = sigma * rng.standard_normal((m, task.d)) / np.sqrt(task.d)
        a = sigma * rng.standard_normal(m) / np.sqrt(m)
        self.w = np.concatenate([W.ravel(), a])

    def unpack(self, w):
        m, d = self.m, self.d
        return w[: m * d].reshape(m, d), w[m * d :]

    def forward(self, w, X):
        W, a = self.unpack(w)
        Hd = np.tanh(X @ W.T)
        return Hd @ a, Hd

    def loss(self, w, X=None, y=None):
        X = self.task.X if X is None else X
        y = self.task.y if y is None else y
        f, _ = self.forward(w, X)
        return 0.5 * np.mean((f - y) ** 2)

    def grad(self, w):
        """Analytic gradient of L = (1/2n) sum (f - y)^2."""
        X, y = self.task.X, self.task.y
        n = X.shape[0]
        W, a = self.unpack(w)
        Z = X @ W.T
        Hd = np.tanh(Z)
        r = (Hd @ a - y) / n                 # (n,)
        ga = Hd.T @ r                        # (m,)
        dZ = np.outer(r, a) * (1.0 - Hd**2)  # (n,m)
        gW = dZ.T @ X                        # (m,d)
        return np.concatenate([gW.ravel(), ga])

    def hvp(self, w, v, eps=1e-5):
        """Hessian-vector product by symmetric finite differences of the gradient."""
        nv = np.linalg.norm(v)
        if nv == 0.0:
            return np.zeros_like(v)
        u = v / nv
        return (self.grad(w + eps * u) - self.grad(w - eps * u)) * nv / (2.0 * eps)

    def sharpness(self, w, v0=None, iters=25):
        """Largest Hessian eigenvalue by warm-started power iteration."""
        v = np.random.default_rng(0).standard_normal(w.size) if v0 is None else v0.copy()
        v /= np.linalg.norm(v)
        S = 0.0
        for _ in range(iters):
            Hv = self.hvp(w, v)
            S = float(v @ Hv)
            nrm = np.linalg.norm(Hv)
            if nrm == 0.0:
                break
            v = Hv / nrm
        return S, v


# ----------------------------------------------------------------------------
# The five controllers
# ----------------------------------------------------------------------------
# Shared convention: each takes the current (eta, S) and a target lam_star,
# and returns the next eta. lam is computed as log|1 - eta*S|, floored so the
# eta*S -> 1 singularity cannot produce -inf.

LAM_FLOOR = -30.0


def _lam(eta, S):
    return max(np.log(abs(1.0 - eta * S) + 1e-300), LAM_FLOOR)


def method1_inversion(eta, S, lam_star=0.0, **kw):
    """I - Direct criticality inversion.

    Solve log|1 - eta*S| = lam* on the negative branch (1 - eta*S = -e^{lam*}):
        eta = (1 + e^{lam*}) / S        ->  eta = 2/S at lam* = 0.
    One-shot, no memory: eta jumps wherever the noisy S measurement points.
    """
    return (1.0 + np.exp(lam_star)) / S


def method2_thermostat(eta, S, lam_star=0.0, K=0.02, **kw):
    """II - Multiplicative ("thermostat") feedback.

        eta_{t+1} = eta_t * exp[K (lam* - lam_t)]

    Proportional control in log-eta space. K is the gain.
    """
    return eta * np.exp(K * (lam_star - _lam(eta, S)))


def method3_distance(eta, S, lam_star=0.0, K=0.05, **kw):
    """III - Distance-to-criticality controller.

        eps_t   = 1 - eta_t S_t / 2
        eps*    = -lam* / 2                  (from lam = -2 eps + O(eps^2))
        eta_{t+1} = eta_t [1 + K (eps_t - eps*)]

    At lam* = 0 this is exactly the boxed formula in the notes, with fixed
    point eta*S = 2. Only needs S, never lam.
    """
    eps = 1.0 - eta * S / 2.0
    return eta * (1.0 + K * (eps - (-lam_star / 2.0)))


def method4_predictive(eta, S, lam_star=0.0, alpha=1.0, **kw):
    """IV - Predictive (Levenberg-damped Gauss-Newton) control.

        g = dlam/deta = -S / (1 - eta S)
        eta_{t+1} = eta_t - (lam_t - lam*) g / (g^2 + alpha)

    Minimises J(eta) = (lam(eta) - lam*)^2 + alpha (eta - eta_t)^2 to first order.
    """
    denom = 1.0 - eta * S
    g = -S / (denom if abs(denom) > 1e-12 else np.sign(denom or 1.0) * 1e-12)
    return eta - (_lam(eta, S) - lam_star) * g / (g * g + alpha)


def method5_ode(eta, S, lam_star=0.0, K=0.05, dt=1.0, **kw):
    """V - First-order relaxation of lam, discretised with step dt.

        dlam/dt = -K (lam - lam*)   =>   deta/dt = K (1 - eta S)/S (lam - lam*)
    """
    return eta + dt * K * (1.0 - eta * S) / S * (_lam(eta, S) - lam_star)


CONTROLLERS = {
    "I_inversion":   method1_inversion,
    "II_thermostat": method2_thermostat,
    "III_distance":  method3_distance,
    "IV_predictive": method4_predictive,
    "V_ode":         method5_ode,
}

# Methods IV and V contain a factor whose sign flips at eta*S = 1, so their
# feedback is INVERTED in the ordered phase. Gate them until eta*S > 1.
NEEDS_GATE = {"IV_predictive", "V_ode"}


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

def run(method=None, m=64, eta0=0.4, lam_star=0.0, steps=3000, seed=0,
        cadence=5, power_iters=25, sigma=1.0, eta_bounds=(1e-4, 10.0),
        gate=True, **ctrl_kw):
    """Train with full-batch GD; optionally adapt eta with one of the controllers.

    method=None gives the paper's uncontrolled baseline (fixed eta).
    Returns a dict of time series.
    """
    task = Task(seed=seed)
    net = Student(task, m=m, sigma=sigma, seed=seed)
    w, eta, v = net.w, float(eta0), None
    ctrl = CONTROLLERS[method] if method else None

    log = {k: [] for k in ("t", "L", "S", "eta", "lam", "etaS")}
    diverged = False

    for t in range(steps):
        if t % cadence == 0:
            S, v = net.sharpness(w, v, iters=(60 if t == 0 else power_iters))
            S = max(S, 1e-8)
            log["t"].append(t)
            log["L"].append(net.loss(w))
            log["S"].append(S)
            log["eta"].append(eta)
            log["lam"].append(_lam(eta, S))
            log["etaS"].append(eta * S)

            if ctrl is not None:
                if gate and method in NEEDS_GATE and eta * S < 1.0:
                    eta = method3_distance(eta, S, lam_star, K=0.05)
                else:
                    eta = ctrl(eta, S, lam_star, **ctrl_kw)
                eta = float(np.clip(eta, *eta_bounds))

        w = w - eta * net.grad(w)
        if not np.isfinite(w).all() or net.loss(w) > 1e8:
            diverged = True
            break

    out = {k: np.asarray(v_) for k, v_ in log.items()}
    out["diverged"] = diverged
    out["test_error"] = (np.nan if diverged
                         else float(net.loss(w, task.Xte, task.yte)))
    # lambda_top = time average over the second half of the run
    half = len(out["lam"]) // 2
    out["lam_top"] = float(np.mean(out["lam"][half:])) if not diverged else np.nan
    return out


if __name__ == "__main__":
    for name in ["baseline"] + list(CONTROLLERS):
        r = run(None if name == "baseline" else name, steps=2000, eta0=0.3)
        tag = "DIVERGED" if r["diverged"] else f"test={r['test_error']:.4f}"
        print(f"{name:16s} lam_top={r['lam_top']:+.4f}  "
              f"final etaS={r['etaS'][-1]:.3f}  eta={r['eta'][-1]:.3f}  {tag}")
