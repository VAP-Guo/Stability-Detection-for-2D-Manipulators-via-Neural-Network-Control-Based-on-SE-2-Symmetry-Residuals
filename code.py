import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.neural_network import MLPRegressor, MLPClassifier # type: ignore
from sklearn.metrics import roc_auc_score # type: ignore

# -----------------------------
# Geometry helpers
# -----------------------------
def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def transform_pose(pose, R, t):
    p = R @ pose[:2] + t
    phi = wrap_angle(pose[2] + np.arctan2(R[1, 0], R[0, 0]))
    return np.array([p[0], p[1], phi])

# -----------------------------
# 2-link planar arm
# -----------------------------
L1, L2 = 1.0, 0.8

def fk(q):
    q1, q2 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    phi = wrap_angle(q1 + q2)
    return np.array([x, y, phi])

def ik(x, y, elbow=1.0):
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = np.clip(c2, -1.0, 1.0)
    q2 = elbow * np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([wrap_angle(q1), wrap_angle(q2)])

def jacobian(q):
    q1, q2 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [ L1 * c1 + L2 * c12,  L2 * c12]
    ])

# -----------------------------
# Target trajectory
# -----------------------------
def target_pose(t, T):
    s = t / max(T - 1, 1)
    ang = 2 * np.pi * s
    x = 0.95 + 0.22 * np.cos(ang)
    y = 0.25 + 0.22 * np.sin(ang)
    dx = -0.22 * 2 * np.pi * np.sin(ang) / max(T - 1, 1)
    dy =  0.22 * 2 * np.pi * np.cos(ang) / max(T - 1, 1)
    phi = np.arctan2(dy, dx)
    return np.array([x, y, wrap_angle(phi)])

# -----------------------------
# Expert policy and controller
# -----------------------------
def expert_policy(obs_pose, tgt_pose):
    e = tgt_pose[:2] - obs_pose[:2]
    u = 1.8 * e
    return np.clip(u, -0.35, 0.35)

def make_controller(X, Y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=0
    )
    model.fit(Xs, Y)
    return model, scaler

def predict_u(model, scaler, obs_vec):
    return model.predict(scaler.transform(obs_vec.reshape(1, -1)))[0]

# -----------------------------
# Episode rollout
# -----------------------------
def simulate_episode(mode, model, scaler, seed=0, T=360, dt=0.02, onset=180):
    rng = np.random.default_rng(seed)

    tgt0 = target_pose(0, T)
    q = ik(tgt0[0], tgt0[1], elbow=1.0)
    qd = np.zeros(2)

    if mode == "drift":
        obs_bias_xy = np.array([0.06, -0.04])
        obs_bias_phi = 0.08
    else:
        obs_bias_xy = np.zeros(2)
        obs_bias_phi = 0.0

    logs = {
        "pose": [],
        "obs_pose": [],
        "tgt_pose": [],
        "u": [],
        "current": [],
        "rsym": [],
        "rinv_feat": [],
        "rresp": [],
        "raw_feat": [],
        "fault": []
    }

    prev_pose = fk(q)

    for t in range(T):
        true_pose = fk(q)
        tgt = target_pose(t, T)

        obs_pose = true_pose.copy()
        if mode == "drift" and t >= onset:
            obs_pose[:2] += obs_bias_xy
            obs_pose[2] = wrap_angle(obs_pose[2] + obs_bias_phi)

        obs_vec = np.r_[obs_pose, tgt]
        if model is None:
            u = expert_policy(obs_pose, tgt)
        else:
            u = predict_u(model, scaler, obs_vec)

        J = jacobian(q)
        qd_cmd = np.linalg.pinv(J) @ u

        gain = 1.0
        disturbance = np.zeros(2)
        if mode == "stall" and t >= onset:
            gain = 0.35
        if mode == "collision" and t >= onset:
            disturbance = np.array([-0.7, 0.45]) * np.exp(-0.01 * (t - onset))

        noise = 0.01 * rng.standard_normal(2)
        qd = 0.90 * qd + gain * qd_cmd + disturbance + noise
        q = wrap_angle(q + dt * qd)

        new_pose = fk(q)
        delta_xy = new_pose[:2] - prev_pose[:2]

        current = 0.25 + 0.65 * np.linalg.norm(qd_cmd) + 1.35 * np.linalg.norm(qd_cmd - qd) + 0.02 * rng.standard_normal()
        current = float(max(current, 0.0))

        thetas = [0.0, 0.20, -0.35]
        rs = 0.0
        for ang in thetas:
            R = rot2(ang)
            txy = np.array([0.05 * np.cos(ang), 0.05 * np.sin(ang)])
            obs_g = transform_pose(obs_pose, R, txy)
            tgt_g = transform_pose(tgt, R, txy)
            if model is None:
                u_g = expert_policy(obs_g, tgt_g)
            else:
                u_g = predict_u(model, scaler, np.r_[obs_g, tgt_g])
            rs += np.linalg.norm(u_g - R @ u)
        rsym = rs / len(thetas)

        dist = np.linalg.norm(tgt[:2] - obs_pose[:2])
        speed = np.linalg.norm(qd)
        rinv_feat = np.array([dist, speed, current])

        rresp = np.linalg.norm(delta_xy - dt * u)

        raw_feat = np.r_[obs_pose, tgt, u, current]
        fault = int((mode != "healthy") and (t >= onset))

        logs["pose"].append(true_pose)
        logs["obs_pose"].append(obs_pose)
        logs["tgt_pose"].append(tgt)
        logs["u"].append(u)
        logs["current"].append(current)
        logs["rsym"].append(rsym)
        logs["rinv_feat"].append(rinv_feat)
        logs["rresp"].append(rresp)
        logs["raw_feat"].append(raw_feat)
        logs["fault"].append(fault)

        prev_pose = new_pose

    for k in logs:
        logs[k] = np.asarray(logs[k])
    return logs

# -----------------------------
# Dataset builders
# -----------------------------
def collect_controller_data(n_eps=25, T=260):
    X, Y = [], []
    for seed in range(n_eps):
        rng = np.random.default_rng(seed)
        tgt0 = target_pose(0, T)
        q = ik(tgt0[0], tgt0[1], elbow=1.0) + 0.12 * rng.standard_normal(2)
        qd = np.zeros(2)
        for t in range(T):
            true_pose = fk(q)
            tgt = target_pose(t, T)
            obs_pose = true_pose.copy()
            obs_vec = np.r_[obs_pose, tgt]
            u = expert_policy(obs_pose, tgt)
            J = jacobian(q)
            qd_cmd = np.linalg.pinv(J) @ u
            qd = 0.90 * qd + qd_cmd + 0.005 * rng.standard_normal(2)
            q = wrap_angle(q + 0.02 * qd)
            X.append(obs_vec)
            Y.append(u)
    return np.asarray(X), np.asarray(Y)

def make_windows(episodes, W=12):
    X, y = [], []
    for ep in episodes:
        feats = ep["raw_feat"]
        labels = ep["fault"]
        for t in range(W - 1, len(feats)):
            X.append(feats[t - W + 1:t + 1].reshape(-1))
            y.append(labels[t])
    return np.asarray(X), np.asarray(y)

# -----------------------------
# Main experiment
# -----------------------------
if __name__ == "__main__":
    # 1) train controller on healthy imitation data
    Xc, Yc = collect_controller_data(n_eps=35, T=260)
    controller, controller_scaler = make_controller(Xc, Yc)

    # 2) train/test episodes
    train_healthy = [simulate_episode("healthy", controller, controller_scaler, seed=10 + i) for i in range(8)]
    train_faults = {
        "collision": [simulate_episode("collision", controller, controller_scaler, seed=40 + i) for i in range(8)],
        "stall":     [simulate_episode("stall", controller, controller_scaler, seed=60 + i) for i in range(8)],
        "drift":     [simulate_episode("drift", controller, controller_scaler, seed=80 + i) for i in range(8)],
    }

    test_healthy = [simulate_episode("healthy", controller, controller_scaler, seed=110 + i) for i in range(6)]
    test_faults = {
        "collision": [simulate_episode("collision", controller, controller_scaler, seed=140 + i) for i in range(6)],
        "stall":     [simulate_episode("stall", controller, controller_scaler, seed=160 + i) for i in range(6)],
        "drift":     [simulate_episode("drift", controller, controller_scaler, seed=180 + i) for i in range(6)],
    }

    # 3) healthy stats for proposed method
    rsym_h = np.concatenate([ep["rsym"] for ep in train_healthy])
    rresp_h = np.concatenate([ep["rresp"] for ep in train_healthy])
    inv_h = np.concatenate([ep["rinv_feat"] for ep in train_healthy], axis=0)
    cur_h = np.concatenate([ep["current"] for ep in train_healthy])

    mu_sym, sd_sym = rsym_h.mean(), rsym_h.std() + 1e-6
    mu_resp, sd_resp = rresp_h.mean(), rresp_h.std() + 1e-6
    mu_inv, sd_inv = inv_h.mean(axis=0), inv_h.std(axis=0) + 1e-6
    mu_cur, sd_cur = cur_h.mean(), cur_h.std() + 1e-6

    def score_episode(ep):
        zsym = np.maximum((ep["rsym"] - mu_sym) / sd_sym, 0.0)
        zresp = np.maximum((ep["rresp"] - mu_resp) / sd_resp, 0.0)
        zinv = np.maximum((ep["rinv_feat"] - mu_inv) / sd_inv, 0.0)
        zinv = np.linalg.norm(zinv, axis=1)
        return 1.00 * zsym + 0.75 * zinv + 1.15 * zresp

    healthy_scores = np.concatenate([score_episode(ep) for ep in train_healthy])
    tau_main = np.percentile(healthy_scores, 95)
    tau_cur = np.percentile(cur_h, 95)

    # 4) autoencoder and temporal classifier baselines
    W = 12
    ae_train_X = []
    for ep in train_healthy:
        feats = ep["raw_feat"]
        for t in range(W - 1, len(feats)):
            ae_train_X.append(feats[t - W + 1:t + 1].reshape(-1))
    ae_train_X = np.asarray(ae_train_X)

    ae_scaler = StandardScaler().fit(ae_train_X)
    ae_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=400,
        random_state=1
    )
    ae_model.fit(ae_scaler.transform(ae_train_X), ae_train_X)

    clf_train_eps = train_healthy + train_faults["collision"] + train_faults["stall"] + train_faults["drift"]
    clf_train_X, clf_train_y = make_windows(clf_train_eps, W=W)
    clf_scaler = StandardScaler().fit(clf_train_X)
    clf_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=400,
        random_state=2
    )
    clf_model.fit(clf_scaler.transform(clf_train_X), clf_train_y)

    # 5) test scores
    def eval_scores(episodes):
        y_all, s_main, s_cur, s_ae, s_clf = [], [], [], [], []
        for ep in episodes:
            y = ep["fault"]
            s_main.append(score_episode(ep))
            s_cur.append(ep["current"])

            feats = ep["raw_feat"]
            ae_s, clf_s = [], []
            for t in range(len(feats)):
                if t < W - 1:
                    ae_s.append(np.nan)
                    clf_s.append(np.nan)
                    continue
                xw = feats[t - W + 1:t + 1].reshape(-1)
                xw_s = ae_scaler.transform(xw.reshape(1, -1))
                recon = ae_model.predict(xw_s)[0]
                ae_s.append(np.mean((recon - xw) ** 2))

                pr = clf_model.predict_proba(clf_scaler.transform(xw.reshape(1, -1)))[0, 1]
                clf_s.append(pr)

            s_ae.append(np.asarray(ae_s))
            s_clf.append(np.asarray(clf_s))
            y_all.append(y)

        y_all = np.concatenate(y_all)
        s_main = np.concatenate(s_main)
        s_cur = np.concatenate(s_cur)
        s_ae = np.concatenate(s_ae)
        s_clf = np.concatenate(s_clf)
        return y_all, s_main, s_cur, s_ae, s_clf

    test_eps = test_healthy + test_faults["collision"] + test_faults["stall"] + test_faults["drift"]
    y_test, s_main, s_cur, s_ae, s_clf = eval_scores(test_eps)

    valid_ae = ~np.isnan(s_ae)
    valid_clf = ~np.isnan(s_clf)

    def auc_safe(y, s):
        m = np.isfinite(s)
        if len(np.unique(y[m])) < 2:
            return np.nan
        return roc_auc_score(y[m], s[m])

    tau_ae = np.percentile(s_ae[(y_test == 0) & valid_ae], 95)
    tau_clf = 0.5

    def episode_summary(episodes, scorer, tau, windowed=False):
        delays, fpr_list, fnr_list = [], [], []
        for ep in episodes:
            y = ep["fault"]
            s = scorer(ep)
            s = s[W - 1:] if windowed else s
            yv = y[W - 1:] if windowed else y
            if yv.max() == 0:
                fpr_list.append(float((s > tau).any()))
            else:
                onset = np.where(yv == 1)[0][0]
                alarms = np.where(s[onset:] > tau)[0]
                if len(alarms) == 0:
                    fnr_list.append(1.0)
                else:
                    fnr_list.append(0.0)
                    delays.append(int(alarms[0]))
        return np.nanmean(delays) if delays else np.nan, np.mean(fpr_list) if fpr_list else np.nan, np.mean(fnr_list) if fnr_list else np.nan

    def main_scorer(ep): return score_episode(ep)
    def cur_scorer(ep): return ep["current"]
    def ae_scorer(ep):
        feats = ep["raw_feat"]
        scores = np.full(len(feats), np.nan)
        for t in range(W - 1, len(feats)):
            xw = feats[t - W + 1:t + 1].reshape(-1)
            recon = ae_model.predict(ae_scaler.transform(xw.reshape(1, -1)))[0]
            scores[t] = np.mean((recon - xw) ** 2)
        return scores
    def clf_scorer(ep):
        feats = ep["raw_feat"]
        scores = np.full(len(feats), np.nan)
        for t in range(W - 1, len(feats)):
            xw = feats[t - W + 1:t + 1].reshape(-1)
            scores[t] = clf_model.predict_proba(clf_scaler.transform(xw.reshape(1, -1)))[0, 1]
        return scores

    auc_main = auc_safe(y_test, s_main)
    auc_cur = auc_safe(y_test, s_cur)
    auc_ae = auc_safe(y_test[valid_ae], s_ae[valid_ae])
    auc_clf = auc_safe(y_test[valid_clf], s_clf[valid_clf])

    d_main, fpr_main, fnr_main = episode_summary(test_eps, main_scorer, tau_main, windowed=False)
    d_cur, fpr_cur, fnr_cur = episode_summary(test_eps, cur_scorer, tau_cur, windowed=False)
    d_ae, fpr_ae, fnr_ae = episode_summary(test_eps, ae_scorer, tau_ae, windowed=True)
    d_clf, fpr_clf, fnr_clf = episode_summary(test_eps, clf_scorer, tau_clf, windowed=True)

    # 6) ablation
    def ablation_score(ep, which):
        zsym = np.maximum((ep["rsym"] - mu_sym) / sd_sym, 0.0)
        zresp = np.maximum((ep["rresp"] - mu_resp) / sd_resp, 0.0)
        zinv = np.maximum((ep["rinv_feat"] - mu_inv) / sd_inv, 0.0)
        zinv = np.linalg.norm(zinv, axis=1)
        if which == "sym":
            return zsym
        if which == "inv":
            return zinv
        if which == "resp":
            return zresp
        return 1.00 * zsym + 0.75 * zinv + 1.15 * zresp

    def collect_auc(which):
        ys, ss = [], []
        for ep in test_eps:
            ys.append(ep["fault"])
            ss.append(ablation_score(ep, which))
        ys = np.concatenate(ys)
        ss = np.concatenate(ss)
        return roc_auc_score(ys, ss)

    auc_sym = collect_auc("sym")
    auc_inv = collect_auc("inv")
    auc_resp = collect_auc("resp")
    auc_fused = collect_auc("fused")

    print("\n=== Overall AUROC ===")
    print(f"Proposed fused score : {auc_fused:.3f}")
    print(f"  rsym only          : {auc_sym:.3f}")
    print(f"  rinv only          : {auc_inv:.3f}")
    print(f"  rresp only         : {auc_resp:.3f}")
    print(f"Current threshold    : {auc_cur:.3f}")
    print(f"Autoencoder          : {auc_ae:.3f}")
    print(f"Temporal classifier  : {auc_clf:.3f}")

    print("\n=== Episode-level summary ===")
    print(f"Proposed fused score : delay={d_main:.2f}, FPR={fpr_main:.3f}, FNR={fnr_main:.3f}")
    print(f"Current threshold    : delay={d_cur:.2f}, FPR={fpr_cur:.3f}, FNR={fnr_cur:.3f}")
    print(f"Autoencoder          : delay={d_ae:.2f}, FPR={fpr_ae:.3f}, FNR={fnr_ae:.3f}")
    print(f"Temporal classifier  : delay={d_clf:.2f}, FPR={fpr_clf:.3f}, FNR={fnr_clf:.3f}")

    # 7) visualize one representative collision episode
    demo = test_faults["collision"][0]
    s_demo = score_episode(demo)
    t = np.arange(len(demo["fault"])) * 0.02

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(t, s_demo, label="S_t")
    ax[0].axhline(tau_main, linestyle="--", label="threshold")
    ax[0].set_ylabel("score")
    ax[0].legend()

    ax[1].plot(t, demo["rsym"], label="r_sym")
    ax[1].plot(t, np.linalg.norm((demo["rinv_feat"] - mu_inv) / sd_inv, axis=1), label="r_inv")
    ax[1].plot(t, demo["rresp"], label="r_resp")
    ax[1].axvline(180 * 0.02, linestyle=":", label="fault onset")
    ax[1].set_ylabel("residual")
    ax[1].legend()

    pose = demo["pose"]
    tgt = demo["tgt_pose"]
    ax[2].plot(pose[:, 0], pose[:, 1], label="end-effector")
    ax[2].plot(tgt[:, 0], tgt[:, 1], label="target")
    ax[2].axis("equal")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[2].legend()

    plt.tight_layout()
    plt.show()