"""Built-in Phase 0 templates for `kaggle_tabular` and `rl_control` domains."""

import json
import textwrap
from frame.schemas import (
    BenchSpec,
    ExternalAnchor,
    GuardrailSpec,
    StatisticalEstimator,
    TargetDirection,
)


def get_kaggle_tabular_template(topic: str) -> dict[str, str]:
    spec = BenchSpec(
        benchmark_id=topic,
        title=f"Kaggle Tabular Benchmark: {topic}",
        domain="kaggle_tabular",
        primary_metric_name="rmse",
        direction=TargetDirection.MINIMIZE,
        estimator=StatisticalEstimator.PAIRED_T_ROPE,
        rope_epsilon=0.06,
        blend_rope_epsilon=0.02,
        vault_overfit_tolerance=0.08,
        pilot_seeds=[101, 102, 103],
        confirmatory_k=5,
        timeout_sec=30.0,
        external_anchor=ExternalAnchor(
            source_name="Synthetic Friedman-Interaction Tabular Reference",
            reference_url_or_citation="Friedman (1991) Multivariate Adaptive Regression Splines",
            competence_floor=1.25,
            published_sota_target=0.25,
        ),
        guardrails=[
            GuardrailSpec(
                name="variance_ratio",
                op=">=",
                threshold=0.55,
                description="Predicted std / target std must not collapse to the mean",
            ),
            GuardrailSpec(
                name="max_abs_error",
                op="<=",
                threshold=5.00,
                description="Tail error cap on extreme samples",
            ),
            GuardrailSpec(
                name="inference_latency_ms",
                op="<=",
                threshold=250.0,
                description="Per-fold prediction latency cap in milliseconds",
            ),
        ],
    )

    harness_code = textwrap.dedent(
        """
        import numpy as np

        def _generate_dataset(seed: int, n_samples: int):
            rng = np.random.default_rng(seed)
            X = rng.uniform(-2.0, 2.0, size=(n_samples, 8))
            # Ground-truth nonlinear interaction surface + regime shift on X[:, 0] > 0.5
            signal = (
                1.2 * np.sin(np.pi * X[:, 0] * X[:, 1])
                + 0.8 * (X[:, 2] - 0.5) ** 2
                + 0.6 * X[:, 3]
                - 0.4 * X[:, 4]
                + np.where(X[:, 0] > 0.5, 0.9 * X[:, 1] * X[:, 2], 0.0)
            )
            y = signal + rng.normal(0.0, 0.15, size=n_samples)
            return X, y

        def prepare_split(seed: int, split_tier: str) -> tuple[dict, dict]:
            tier_offset = {"smoke": 1, "dev_diagnostic": 100, "confirmatory": 10_000, "vault_holdout": 90_000}[split_tier]
            n_train = 120 if split_tier == "smoke" else 700
            n_eval = 60 if split_tier == "smoke" else 300
            X, y = _generate_dataset(seed + tier_offset, n_train + n_eval)
            X_train, y_train = X[:n_train], y[:n_train]
            X_eval, y_eval = X[n_train:], y[n_train:]

            public_task = {
                "X_train": X_train,
                "y_train": y_train,
                "X_eval": X_eval,
                "feature_names": [f"f_{i}" for i in range(X.shape[1])],
            }
            secret_targets = {
                "y_eval": y_eval,
                "X_eval": X_eval,
                "y_train_mean": float(np.mean(y_train)),
            }
            return public_task, secret_targets

        def score_candidate_output(candidate_output: dict, secret_targets: dict, seed: int, split_tier: str) -> dict:
            preds = np.asarray(candidate_output["predictions"], dtype=np.float64).reshape(-1)
            y_true = np.asarray(secret_targets["y_eval"], dtype=np.float64).reshape(-1)
            X_eval = np.asarray(secret_targets["X_eval"], dtype=np.float64)
            if preds.shape != y_true.shape:
                raise ValueError(f"Prediction shape mismatch: expected {y_true.shape}, got {preds.shape}")
            if not np.all(np.isfinite(preds)):
                raise ValueError("Predictions contain NaN or Inf.")

            residuals = y_true - preds
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            std_y = max(float(np.std(y_true)), 1e-8)
            var_ratio = float(np.std(preds) / std_y)
            max_err = float(np.max(np.abs(residuals)))
            latency_ms = float(candidate_output.get("inference_latency_ms", 1.0))

            diagnostics = {}
            if split_tier == "dev_diagnostic":
                high_f0_mask = X_eval[:, 0] > 0.5
                diagnostics = {
                    "overall_rmse": round(rmse, 4),
                    "regime_f0_gt_0.5_rmse": round(float(np.sqrt(np.mean(residuals[high_f0_mask] ** 2))), 4),
                    "regime_f0_le_0.5_rmse": round(float(np.sqrt(np.mean(residuals[~high_f0_mask] ** 2))), 4),
                    "residual_q90_abs": round(float(np.quantile(np.abs(residuals), 0.90)), 4),
                    "hint": "Errors are concentrated in nonlinear interaction terms (f_0 * f_1) and regime f_0 > 0.5.",
                }

            return {
                "primary_metric": rmse,
                "guardrails": {
                    "variance_ratio": var_ratio,
                    "max_abs_error": max_err,
                    "inference_latency_ms": latency_ms,
                },
                "diagnostics": diagnostics,
            }

        def generate_null_outputs(public_task: dict, secret_targets: dict, seed: int) -> dict[str, dict]:
            rng = np.random.default_rng(seed + 555)
            y_mean = float(secret_targets["y_train_mean"])
            n = secret_targets["y_eval"].shape[0]
            y_train = np.asarray(public_task["y_train"], dtype=np.float64)
            return {
                "constant_mean": {"predictions": np.full(n, y_mean), "inference_latency_ms": 0.1},
                "zero_predictor": {"predictions": np.zeros(n), "inference_latency_ms": 0.1},
                "random_permutation": {"predictions": rng.choice(y_train, size=n, replace=True), "inference_latency_ms": 0.1},
                "gaussian_noise": {"predictions": rng.normal(y_mean, float(np.std(y_train)), size=n), "inference_latency_ms": 0.1},
            }

        def degrade_output(candidate_output: dict, secret_targets: dict, noise_level: float, seed: int) -> dict:
            rng = np.random.default_rng(seed + int(noise_level * 1000))
            preds = np.asarray(candidate_output["predictions"], dtype=np.float64).copy()
            scale = float(np.std(secret_targets["y_eval"])) * (1.8 * noise_level)
            noisy_preds = preds + rng.normal(0.0, scale, size=preds.shape)
            return {"predictions": noisy_preds, "inference_latency_ms": candidate_output.get("inference_latency_ms", 1.0)}
        """
    ).strip() + "\n"

    baseline_code = textwrap.dedent(
        """
        from pathlib import Path
        import time
        import numpy as np
        from sklearn.linear_model import Ridge

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            X_train = np.asarray(task["X_train"], dtype=np.float64)
            y_train = np.asarray(task["y_train"], dtype=np.float64)
            X_eval = np.asarray(task["X_eval"], dtype=np.float64)

            # Competent Gen-0 baseline with basic quadratic terms
            X_tr_feat = np.hstack([X_train, X_train[:, :3] ** 2, (X_train[:, 0] * X_train[:, 1])[:, None]])
            X_ev_feat = np.hstack([X_eval, X_eval[:, :3] ** 2, (X_eval[:, 0] * X_eval[:, 1])[:, None]])

            model = Ridge(alpha=1.0, random_state=seed)
            model.fit(X_tr_feat, y_train)

            t0 = time.perf_counter()
            preds = model.predict(X_ev_feat)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "predictions": preds,
                "inference_latency_ms": latency_ms,
            }
        """
    ).strip() + "\n"

    positive_control_code = textwrap.dedent(
        """
        from pathlib import Path
        import time
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            X_train = np.asarray(task["X_train"], dtype=np.float64)
            y_train = np.asarray(task["y_train"], dtype=np.float64)
            X_eval = np.asarray(task["X_eval"], dtype=np.float64)

            def _build_features(X: np.ndarray) -> np.ndarray:
                f0_f1 = (X[:, 0] * X[:, 1])[:, None]
                f1_f2 = (X[:, 1] * X[:, 2])[:, None]
                sin_f01 = np.sin(np.pi * X[:, 0] * X[:, 1])[:, None]
                return np.hstack([X, f0_f1, f1_f2, sin_f01])

            model = HistGradientBoostingRegressor(
                max_iter=(35 if mode == "smoke" else 120),
                learning_rate=0.08,
                max_depth=5,
                random_state=seed,
            )
            model.fit(_build_features(X_train), y_train)

            t0 = time.perf_counter()
            preds = model.predict(_build_features(X_eval))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "predictions": preds,
                "inference_latency_ms": latency_ms,
            }
        """
    ).strip() + "\n"

    return {
        "bench_spec.json": json.dumps(spec.model_dump(mode="json"), indent=2),
        "harness.py": harness_code,
        "baseline.py": baseline_code,
        "positive_control.py": positive_control_code,
    }


def get_rl_control_template(topic: str) -> dict[str, str]:
    spec = BenchSpec(
        benchmark_id=topic,
        title=f"RL Continuous Control Benchmark: {topic}",
        domain="rl_control",
        primary_metric_name="mean_episode_return",
        direction=TargetDirection.MAXIMIZE,
        estimator=StatisticalEstimator.IQM_BOOTSTRAP_ROPE,
        rope_epsilon=0.50,
        blend_rope_epsilon=0.25,
        vault_overfit_tolerance=4.00,
        pilot_seeds=[101, 102, 103],
        confirmatory_k=5,
        timeout_sec=30.0,
        external_anchor=ExternalAnchor(
            source_name="Nonlinear Aerodynamic Wind Pendulum Reference",
            reference_url_or_citation="CleanRL / Continuous Control Benchmark Standard",
            competence_floor=-45.0,
            published_sota_target=-12.0,
        ),
        guardrails=[
            GuardrailSpec(
                name="nonfinite_step_count",
                op="<=",
                threshold=0.0,
                description="Zero non-finite or unstable rollout steps allowed",
            ),
            GuardrailSpec(
                name="terminal_state_error",
                op="<=",
                threshold=2.00,
                description="Mean final step distance from equilibrium",
            ),
            GuardrailSpec(
                name="tail_displacement_return",
                op=">=",
                threshold=-120.0,
                description="Mean return on episodes with large initial displacement |x_0| > 1.2",
            ),
            GuardrailSpec(
                name="perturbed_plant_return",
                op=">=",
                threshold=-80.0,
                description="Mean return under 20% plant parameter perturbation",
            ),
        ],
    )

    harness_code = textwrap.dedent(
        """
        import hashlib
        import os
        import numpy as np

        OBS_DIM = 2
        DT = 0.05
        HORIZON = 50
        U_MAX = 5.0
        COST_WEIGHTS = {"q_x": 0.55, "q_v": 0.22, "r_u": 0.18}
        PLANT_PARAMS = {
            "k_lin": -1.35,
            "c_lin": -0.85,
            "k_cubic": -0.90,
            "k_quad_wind": 3.80,
            "c_shear": -1.20,
            "wind_cap": 5.0,
        }


        def _plant_acceleration(
            x: np.ndarray,
            v: np.ndarray,
            u: np.ndarray,
            noise: np.ndarray,
            nl_scale: float = 1.0,
        ) -> np.ndarray:
            nl_raw = nl_scale * (
                PLANT_PARAMS["k_cubic"] * (x**3)
                + PLANT_PARAMS["k_quad_wind"] * (x**2)
                + PLANT_PARAMS["c_shear"] * (x * v)
            )
            nl_wind = PLANT_PARAMS["wind_cap"] * np.tanh(nl_raw / PLANT_PARAMS["wind_cap"])
            return PLANT_PARAMS["k_lin"] * x + PLANT_PARAMS["c_lin"] * v + nl_wind + u + noise


        def _evaluate_policy_action(candidate_output: dict, x: np.ndarray, v: np.ndarray) -> np.ndarray:
            \"\"\"Evaluates candidate's open `select_action(obs, policy_state)` or legacy weight vector.\"\"\"
            obs = np.stack([x, v], axis=-1)
            fn = candidate_output.get("_select_action")
            if callable(fn):
                raw_u = np.asarray(fn(obs, candidate_output), dtype=np.float64).ravel()
            elif "policy_weights" in candidate_output:
                W = np.asarray(candidate_output["policy_weights"], dtype=np.float64).ravel()
                b = float(np.asarray(candidate_output.get("policy_bias", [0.0]), dtype=np.float64).ravel()[0])
                basis_cols = [x, v, x**3, x**2, x * v, np.tanh(x), np.tanh(v)]
                feats = np.stack(basis_cols[: len(W)], axis=-1)
                raw_u = (feats @ W[: feats.shape[1]]) + b
            else:
                raise KeyError(
                    "Candidate must define `select_action(obs, policy_state)` or provide `'policy_weights'`."
                )

            gain_scale = float(candidate_output.get("_degrade_gain_scale", 1.0))
            act_bias = float(candidate_output.get("_degrade_act_bias", 0.0))
            return gain_scale * raw_u + act_bias


        def _simulate_policy(
            candidate_output: dict,
            episodes_init: np.ndarray,
            noise_seq: np.ndarray,
            nl_scale: float = 1.0,
        ) -> dict:
            n_eps, horizon, _ = noise_seq.shape
            x = episodes_init[:, 0].astype(np.float64).copy()
            v = episodes_init[:, 1].astype(np.float64).copy()
            returns = np.zeros(n_eps, dtype=np.float64)
            nonfinite_count = 0

            for t in range(horizon):
                raw_u = _evaluate_policy_action(candidate_output, x, v)
                if raw_u.shape != (n_eps,):
                    raise ValueError(f"select_action must return shape ({n_eps},), got {raw_u.shape}")

                step_nonfinite = ~np.isfinite(raw_u) | ~np.isfinite(x) | ~np.isfinite(v)
                if np.any(step_nonfinite):
                    nonfinite_count += int(np.sum(step_nonfinite))
                    raw_u = np.nan_to_num(raw_u, nan=0.0, posinf=U_MAX, neginf=-U_MAX)
                    x = np.nan_to_num(x, nan=10.0, posinf=10.0, neginf=-10.0)
                    v = np.nan_to_num(v, nan=10.0, posinf=10.0, neginf=-10.0)

                u = np.clip(raw_u, -U_MAX, U_MAX)
                cost = (
                    COST_WEIGHTS["q_x"] * (x**2)
                    + COST_WEIGHTS["q_v"] * (v**2)
                    + COST_WEIGHTS["r_u"] * (u**2)
                )
                returns -= cost

                acc = _plant_acceleration(x, v, u, noise_seq[:, t, 0], nl_scale=nl_scale)
                x_next = x + DT * v
                v_next = v + DT * acc

                post_nonfinite = (
                    ~np.isfinite(x_next)
                    | ~np.isfinite(v_next)
                    | (np.abs(x_next) > 1e4)
                    | (np.abs(v_next) > 1e4)
                )
                if np.any(post_nonfinite):
                    nonfinite_count += int(np.sum(post_nonfinite))
                    x_next = np.clip(
                        np.nan_to_num(x_next, nan=10.0, posinf=10.0, neginf=-10.0), -25.0, 25.0
                    )
                    v_next = np.clip(
                        np.nan_to_num(v_next, nan=10.0, posinf=10.0, neginf=-10.0), -25.0, 25.0
                    )

                x, v = x_next, v_next

            term_errors = np.sqrt(x**2 + v**2)
            high_disp_mask = np.abs(episodes_init[:, 0]) > 1.2
            high_init_return = (
                float(np.mean(returns[high_disp_mask]))
                if np.any(high_disp_mask)
                else float(np.mean(returns))
            )

            return {
                "mean_return": float(np.mean(returns)),
                "term_error": float(np.mean(term_errors)),
                "high_init_return": high_init_return,
                "nonfinite_step_count": float(nonfinite_count),
            }


        def _derive_rng(seed: int, split_tier: str, stream_name: str) -> np.random.Generator:
            secret_salt = os.environ.get(
                "FRAME_BENCH_SECRET_SALT", "wind_pendulum_v2_default_salt"
            )
            payload = f"{secret_salt}:{split_tier}:{stream_name}:{seed}".encode("utf-8")
            digest = hashlib.blake2b(payload, digest_size=16).digest()
            seed_int = int.from_bytes(digest, byteorder="little")
            return np.random.default_rng(seed_int)


        def prepare_split(seed: int, split_tier: str) -> tuple[dict, dict]:
            if split_tier not in {"smoke", "dev_diagnostic", "confirmatory", "vault_holdout"}:
                raise ValueError(f"Unknown split_tier: {split_tier}")

            horizon = HORIZON
            n_train_eps = 32 if split_tier == "smoke" else 120
            n_eval_eps = 40 if split_tier == "smoke" else 200

            train_rng = _derive_rng(seed, split_tier, "public_train")
            eval_rng = _derive_rng(seed, split_tier, "secret_eval")
            pert_rng = _derive_rng(seed, split_tier, "secret_perturbed")

            train_init = train_rng.uniform(-1.8, 1.8, size=(n_train_eps, 2))
            train_noise = train_rng.normal(0.0, 0.15, size=(n_train_eps, horizon, 1))

            eval_init = eval_rng.uniform(-1.8, 1.8, size=(n_eval_eps, 2))
            eval_noise = eval_rng.normal(0.0, 0.15, size=(n_eval_eps, horizon, 1))
            perturbed_noise = pert_rng.normal(0.0, 0.30, size=(n_eval_eps, horizon, 1))

            public_task = {
                "train_init": train_init,
                "train_noise": train_noise,
                "obs_dim": OBS_DIM,
                "horizon": horizon,
                "dt": DT,
                "u_max": U_MAX,
                "cost_weights": dict(COST_WEIGHTS),
                "plant_params": dict(PLANT_PARAMS),
            }
            secret_targets = {
                "eval_init": eval_init,
                "eval_noise": eval_noise,
                "perturbed_noise": perturbed_noise,
            }
            return public_task, secret_targets


        def score_candidate_output(
            candidate_output: dict, secret_targets: dict, seed: int, split_tier: str
        ) -> dict:
            sim = _simulate_policy(
                candidate_output, secret_targets["eval_init"], secret_targets["eval_noise"], nl_scale=1.0
            )
            pert_sim = _simulate_policy(
                candidate_output,
                secret_targets["eval_init"],
                secret_targets["perturbed_noise"],
                nl_scale=1.20,
            )

            total_nonfinite = sim["nonfinite_step_count"] + pert_sim["nonfinite_step_count"]

            diagnostics = {}
            if split_tier == "dev_diagnostic":
                diagnostics = {
                    "mean_episode_return": round(sim["mean_return"], 4),
                    "tail_displacement_return": round(sim["high_init_return"], 4),
                    "perturbed_plant_return": round(pert_sim["mean_return"], 4),
                    "terminal_state_error": round(sim["term_error"], 4),
                    "nonfinite_step_count": round(total_nonfinite, 4),
                }

            return {
                "primary_metric": sim["mean_return"],
                "guardrails": {
                    "nonfinite_step_count": total_nonfinite,
                    "terminal_state_error": sim["term_error"],
                    "tail_displacement_return": sim["high_init_return"],
                    "perturbed_plant_return": pert_sim["mean_return"],
                },
                "diagnostics": diagnostics,
            }


        def generate_null_outputs(
            public_task: dict, secret_targets: dict, seed: int
        ) -> dict[str, dict]:
            rng = _derive_rng(seed, "null_probes", "null_generator")
            rand_w = rng.normal(0.0, 0.5, size=2)
            return {
                "zero_controller": {"_select_action": lambda obs, ps: np.zeros(len(obs), dtype=np.float64)},
                "destabilizing_positive_feedback": {
                    "_select_action": lambda obs, ps: 1.2 * obs[:, 0] + 0.8 * obs[:, 1]
                },
                "random_gaussian_policy": {
                    "_select_action": lambda obs, ps: obs @ rand_w
                },
                "constant_bang_bias": {
                    "_select_action": lambda obs, ps: np.full(len(obs), U_MAX * 0.95, dtype=np.float64)
                },
            }


        def degrade_output(
            candidate_output: dict, secret_targets: dict, noise_level: float, seed: int
        ) -> dict:
            \"\"\"Architecture-agnostic actuator degradation wrapper (works on any policy_state representation).\"\"\"
            degraded = dict(candidate_output)
            degraded["_degrade_gain_scale"] = max(0.0, 1.0 - 1.50 * noise_level)
            degraded["_degrade_act_bias"] = 2.20 * noise_level
            return degraded
        """
    ).strip() + "\n"

    baseline_code = textwrap.dedent(
        """
        from pathlib import Path
        import numpy as np
        from scipy.linalg import solve_discrete_are
        from scipy.optimize import minimize


        def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:
            \"\"\"Evaluates policy control action u for batch of states obs = [x, v] of shape (N, 2).\"\"\"
            x = obs[:, 0]
            v = obs[:, 1]
            k_lin = np.asarray(policy_state["k_lin"], dtype=np.float64)
            return k_lin[0] * x + k_lin[1] * v


        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            \"\"\"
            Competent Linear Anchor (Node 0):
            Solves exact Discrete-Time Algebraic Riccati Equation (DARE) for the linearized plant at (0, 0)
            and refines the 2-parameter linear feedback [k_x, k_v] on the training split.
            Note: Does not yet compensate for nonlinear aerodynamic lift (+k_quad_wind * x^2),
            cubic stiffness (k_cubic * x^3), cross-wind shear (c_shear * x * v), or tanh wind saturation.
            \"\"\"
            dt = float(task["dt"])
            u_max = float(task["u_max"])
            cw = task["cost_weights"]
            pp = task["plant_params"]
            train_init = task["train_init"]
            train_noise = task["train_noise"]

            A = np.array(
                [[1.0, dt], [pp["k_lin"] * dt, 1.0 + pp["c_lin"] * dt]], dtype=np.float64
            )
            B = np.array([[0.0], [dt]], dtype=np.float64)
            Q = np.diag([cw["q_x"], cw["q_v"]])
            R = np.array([[cw["r_u"]]], dtype=np.float64)
            P = solve_discrete_are(A, B, Q, R)
            K_lqr = (-np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)).reshape(2)

            def sim_linear(w_lin: np.ndarray) -> float:
                x = train_init[:, 0].astype(np.float64).copy()
                v = train_init[:, 1].astype(np.float64).copy()
                total_cost = np.zeros(len(x), dtype=np.float64)
                for t in range(train_noise.shape[1]):
                    u = np.clip(w_lin[0] * x + w_lin[1] * v, -u_max, u_max)
                    total_cost += cw["q_x"] * (x**2) + cw["q_v"] * (v**2) + cw["r_u"] * (u**2)
                    nl_raw = (
                        pp["k_cubic"] * (x**3)
                        + pp["k_quad_wind"] * (x**2)
                        + pp["c_shear"] * (x * v)
                    )
                    nl_wind = pp["wind_cap"] * np.tanh(nl_raw / pp["wind_cap"])
                    acc = pp["k_lin"] * x + pp["c_lin"] * v + nl_wind + u + train_noise[:, t, 0]
                    x = x + dt * v
                    v = v + dt * acc
                return float(np.mean(total_cost))

            opt = minimize(
                sim_linear,
                x0=K_lqr,
                method="Nelder-Mead",
                options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-3},
            )
            best_lin = opt.x if opt.fun < sim_linear(K_lqr) else K_lqr

            return {"k_lin": np.asarray(best_lin, dtype=np.float64)}
        """
    ).strip() + "\n"

    positive_control_code = textwrap.dedent(
        """
        from pathlib import Path
        import numpy as np
        from scipy.linalg import solve_discrete_are
        from scipy.optimize import minimize


        def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:
            x = obs[:, 0]
            v = obs[:, 1]
            w = np.asarray(policy_state["w_poly"], dtype=np.float64)
            return w[0] * x + w[1] * v + w[2] * (x**3) + w[3] * (x**2)


        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            dt = float(task["dt"])
            u_max = float(task["u_max"])
            cw = task["cost_weights"]
            pp = task["plant_params"]
            train_init = task["train_init"]
            train_noise = task["train_noise"]

            A = np.array(
                [[1.0, dt], [pp["k_lin"] * dt, 1.0 + pp["c_lin"] * dt]], dtype=np.float64
            )
            B = np.array([[0.0], [dt]], dtype=np.float64)
            Q = np.diag([cw["q_x"], cw["q_v"]])
            R = np.array([[cw["r_u"]]], dtype=np.float64)
            P = solve_discrete_are(A, B, Q, R)
            K_lqr = (-np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)).reshape(2)

            w_init = np.array([K_lqr[0], K_lqr[1], -0.45, -1.15], dtype=np.float64)

            def sim_nonlinear(w: np.ndarray) -> float:
                x = train_init[:, 0].astype(np.float64).copy()
                v = train_init[:, 1].astype(np.float64).copy()
                total_cost = np.zeros(len(x), dtype=np.float64)
                for t in range(train_noise.shape[1]):
                    raw_u = w[0] * x + w[1] * v + w[2] * (x**3) + w[3] * (x**2)
                    u = np.clip(raw_u, -u_max, u_max)
                    total_cost += cw["q_x"] * (x**2) + cw["q_v"] * (v**2) + cw["r_u"] * (u**2)
                    nl_raw = (
                        pp["k_cubic"] * (x**3)
                        + pp["k_quad_wind"] * (x**2)
                        + pp["c_shear"] * (x * v)
                    )
                    nl_wind = pp["wind_cap"] * np.tanh(nl_raw / pp["wind_cap"])
                    acc = pp["k_lin"] * x + pp["c_lin"] * v + nl_wind + u + train_noise[:, t, 0]
                    x = x + dt * v
                    v = v + dt * acc
                return float(np.mean(total_cost))

            opt = minimize(
                sim_nonlinear,
                x0=w_init,
                method="Powell",
                options={"maxiter": 120, "ftol": 1e-4},
            )
            W_best = opt.x if opt.fun < sim_nonlinear(w_init) else w_init

            return {"w_poly": np.asarray(W_best, dtype=np.float64)}
        """
    ).strip() + "\n"

    return {
        "bench_spec.json": json.dumps(spec.model_dump(mode="json"), indent=2),
        "harness.py": harness_code,
        "baseline.py": baseline_code,
        "positive_control.py": positive_control_code,
    }
