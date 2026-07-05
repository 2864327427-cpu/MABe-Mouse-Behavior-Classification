import re
import json
import itertools
import numpy as np
import pandas as pd


# %% ==================== 特征工程V0 ====================
def _scale(windows, fps, ref=30.0):
    """
    按FPS线性缩放窗口长度：把以30fps为基准的帧数映射到目标fps。
    例如：基准30帧(1秒)，在60fps视频中应自动调整为60帧，以保证物理时间跨度一致。
    """
    return max(1, int(round(windows * float(fps) / ref)))

def _scale_signed(windows, fps, ref=30.0):
    """
    带符号的帧偏移缩放（正负lag），零特殊处理为0。
    用于 shift 操作：正数表示取过去的数据（lag），负数表示取未来的数据（lead）。
    """
    if windows == 0:
        return 0
    s = 1 if windows > 0 else -1
    mag = max(1, int(round(abs(windows) * float(fps) / ref)))
    return s * mag

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    """
    社交交互特征 (Social interaction features)。
    计算两只老鼠之间的领头-跟随关系、追逐行为及运动相关性。
    """
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    # 相对位置向量 (A -> B)
    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    # 各自的速度分量
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    # 计算 "Lead" (领头程度)：
    # 将 A 的速度向量投影到 (A->B) 的连线上。
    # 正值表示 A 正在朝向 B 运动（或 B 在 A 的前方），负值表示背离。
    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    # 接近速度 (Approach speed): 距离变小的速率
    approach = -rel_dist.diff() 
    # 定义 "Chase" (追逐): 距离正在缩短 (approach > 0) 且 B 正在背离 A 跑 (B_lead > 0，即B的速度方向是远离A的)
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    # 运动速度相关性：两只老鼠是否同时跑或同时停
    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X

# %% ==================== 特征工程 v202 ====================
def add_sniff_proxy_features_pair(
    X: pd.DataFrame,
    mouse_pair: pd.DataFrame,
    avail_A,
    avail_B,
    fps: float,
):
    """
    Sniff 细分的“部位代理 + 动态”显式特征。

    目标：
    - 提供 nose_agent 到 face/body/genital 相关 proxy 的距离与动态
    - 降低模型从 12+ 全笛卡尔距离中“自己找语义组合”的负担

    设计原则：
    - 只在关键点存在时计算
    - 特征数控制在一个小簇，避免无谓膨胀
    """

    def _xy(side: str, part: str):
        try:
            return mouse_pair[side][part]['x'], mouse_pair[side][part]['y']
        except Exception:
            return None, None

    def _dist(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _centroid(side: str):
        # fallback：用该鼠所有可用关键点的质心
        try:
            parts = mouse_pair[side].columns.get_level_values(0).unique()
        except Exception:
            return None, None

        xs, ys = [], []
        for p in parts:
            try:
                xs.append(mouse_pair[side][p]['x'])
                ys.append(mouse_pair[side][p]['y'])
            except Exception:
                continue
        if not xs:
            return None, None
        cx = pd.concat(xs, axis=1).mean(axis=1)
        cy = pd.concat(ys, axis=1).mean(axis=1)
        return cx, cy

    # ===== Agent nose 必须存在 =====
    if 'nose' not in avail_A:
        return X

    Ax, Ay = _xy('A', 'nose')
    if Ax is None:
        return X

    # ===== Target proxies =====
    # Face proxy: ear_mean / nose
    Bnx, Bny = _xy('B', 'nose')
    Belx, Bely = _xy('B', 'ear_left')
    Berx, Bery = _xy('B', 'ear_right')

    ear_mean_x = ear_mean_y = None
    if Belx is not None and Berx is not None:
        ear_mean_x = (Belx + Berx) / 2.0
        ear_mean_y = (Bely + Bery) / 2.0

    # Genital proxy: hip_mean 优先，其次 tail_base
    Bhxl, Bhyl = _xy('B', 'hip_left')
    Bhxr, Bhyr = _xy('B', 'hip_right')
    hip_mean_x = hip_mean_y = None
    if Bhxl is not None and Bhxr is not None:
        hip_mean_x = (Bhxl + Bhxr) / 2.0
        hip_mean_y = (Bhyl + Bhyr) / 2.0

    Btbx, Btby = _xy('B', 'tail_base')

    # Body proxy: body_center / neck / centroid
    Bbcx, Bbcy = _xy('B', 'body_center')
    Bnkx, Bnky = _xy('B', 'neck')
    if Bbcx is not None:
        body_x, body_y = Bbcx, Bbcy
    elif Bnkx is not None:
        body_x, body_y = Bnkx, Bnky
    else:
        body_x, body_y = _centroid('B')

    # ===== 构造若干核心距离 =====
    d_feats = {}

    # nose_A -> nose_B
    if Bnx is not None:
        d_feats['nn'] = _dist(Ax, Ay, Bnx, Bny)

    # nose_A -> ear_left/right/mean
    if Belx is not None:
        d_feats['ne_l'] = _dist(Ax, Ay, Belx, Bely)
    if Berx is not None:
        d_feats['ne_r'] = _dist(Ax, Ay, Berx, Bery)
    if ear_mean_x is not None:
        d_feats['ne_m'] = _dist(Ax, Ay, ear_mean_x, ear_mean_y)

    # nose_A -> tail_base (genital proxy)
    if Btbx is not None:
        d_feats['ntb'] = _dist(Ax, Ay, Btbx, Btby)

    # nose_A -> hip_mean (更贴近生殖区的 proxy)
    if hip_mean_x is not None:
        d_feats['nhm'] = _dist(Ax, Ay, hip_mean_x, hip_mean_y)

    # nose_A -> body proxy
    if body_x is not None:
        d_feats['nbd'] = _dist(Ax, Ay, body_x, body_y)

    if not d_feats:
        return X

    # ===== 写入基础距离 =====
    for k, d in d_feats.items():
        X[f'snf_d_{k}'] = d

    # ===== 聚合距离（face/body/genital） =====
    # face: nose/ears/ear_mean 的最小值
    face_candidates = []
    for kk in ['nn', 'ne_l', 'ne_r', 'ne_m']:
        if kk in d_feats:
            face_candidates.append(d_feats[kk])
    if face_candidates:
        X['snf_dmin_face'] = pd.concat(face_candidates, axis=1).min(axis=1)

    # genital: hip_mean 与 tail_base 的最小值
    gen_candidates = []
    for kk in ['nhm', 'ntb']:
        if kk in d_feats:
            gen_candidates.append(d_feats[kk])
    if gen_candidates:
        X['snf_dmin_genital'] = pd.concat(gen_candidates, axis=1).min(axis=1)

    # body: body proxy
    if 'nbd' in d_feats:
        X['snf_d_body'] = d_feats['nbd']

    # ===== 动态：多 lag 变化（接近率代理） =====
    for lag_base in [5, 10, 20]:
        l = _scale(lag_base, fps)
        for k, d in d_feats.items():
            # 正值：距离变大；负值：距离变小（更“接近”）
            X[f'snf_dch_{k}_l{lag_base}'] = (d - d.shift(l)).fillna(0.0)

    # ===== 动态：rolling mean/std =====
    for w_base in [15, 30, 60]:
        w = _scale(w_base, fps)
        mp = max(1, w // 4)
        for k, d in d_feats.items():
            X[f'snf_dm_{k}_w{w_base}'] = d.rolling(w, min_periods=mp).mean()
            X[f'snf_ds_{k}_w{w_base}'] = d.rolling(w, min_periods=mp).std()

    # ===== 近距离阈值（cm） =====
    # 让树模型更容易做“是否贴近某部位”的分裂
    for k, d in d_feats.items():
        X[f'snf_close_{k}_2cm'] = (d < 2.0).astype(float)
        X[f'snf_close_{k}_5cm'] = (d < 5.0).astype(float)

    return X


# %% ==================== 特征工程 pair v203 gemini3 groom ====================
def add_pair_grooming_features(X, mouse_pair, avail_A, avail_B, fps):
    """
    ===== NEW: Grooming / Dominance Grooming 专用特征 =====
    
    核心思想：
    1. Jitter (微颤): Grooming 包含高频低幅运动。计算加速度的二阶差分能量。
    2. Passivity (被动性): 接收 Grooming 的一方 (B) 通常是静止的。
    3. Relative Jitter (相对微颤): 这是一个强特征。A 的鼻子相对于 B 的身体中心的震动。
       这能剔除两人同时跑动时的共模震动，只保留 A 在 B 身上"动手动脚"的信号。
    """
    
    # 1. 确定身体中心 (用于计算相对运动)
    def _get_centroid(side, avail):
        if 'body_center' in avail:
            return mouse_pair[side]['body_center']['x'], mouse_pair[side]['body_center']['y']
        elif 'neck' in avail: # Neck 也是很好的中心代理
            return mouse_pair[side]['neck']['x'], mouse_pair[side]['neck']['y']
        elif 'tail_base' in avail:
            return mouse_pair[side]['tail_base']['x'], mouse_pair[side]['tail_base']['y']
        else:
            # Fallback: mean of all parts
            xs = mouse_pair[side].xs('x', level=1, axis=1)
            ys = mouse_pair[side].xs('y', level=1, axis=1)
            return xs.mean(axis=1), ys.mean(axis=1)

    Ax_c, Ay_c = _get_centroid('A', avail_A)
    Bx_c, By_c = _get_centroid('B', avail_B)
    
    # 2. 确定动作部位 (对于 Grooming, 主要是 A 的 Nose)
    # 如果没有 Nose, 用 Head/Neck 替代
    if 'nose' in avail_A:
        Ax_n, Ay_n = mouse_pair['A']['nose']['x'], mouse_pair['A']['nose']['y']
    else:
        Ax_n, Ay_n = Ax_c, Ay_c # Fallback to center if no nose

    # === 特征 A: 相对运动的高频震荡 (Relative Jitter) ===
    # 向量 V_rel = A_nose - B_center
    # 我们关心 V_rel 的高频变化，这代表 A 在 B 身上快速移动
    rel_x = Ax_n - Bx_c
    rel_y = Ay_n - By_c
    
    # 计算相对位置的二阶差分 (近似加加速度 Jerk/High Freq Noise)
    # diff(1) 是速度, diff(2) 是加速度, diff(3) 是高频抖动
    # 使用 hypot 合成模长
    rel_jitter = np.sqrt(rel_x.diff().diff()**2 + rel_y.diff().diff()**2).fillna(0.0)
    
    # === 特征 B: 目标 B 的静止程度 (Target Passivity) ===
    # 计算 B 中心的速度
    B_speed = np.sqrt(Bx_c.diff()**2 + By_c.diff()**2).fillna(0.0) * float(fps)
    
    # === 特征 C: 主动方 A 的绝对震动 (Agent Absolute Jitter) ===
    A_jitter = np.sqrt(Ax_n.diff().diff()**2 + Ay_n.diff().diff()**2).fillna(0.0)

    # === 多尺度窗口统计 ===
    # 重点关注短窗口 0.2s - 0.5s (6 - 15 frames @30fps)
    base_windows = [6, 15, 30] 
    
    for bw in base_windows:
        w = _scale(bw, fps)
        mp = max(1, w // 3)
        tag = f"{bw}f" # e.g. 6f, 15f
        
        # 1. 相对震动均值 (A在B身上抖得厉害吗?)
        X[f'rel_jit_m{tag}'] = rel_jitter.rolling(w, min_periods=mp).mean()
        
        # 2. 相对震动最大值 (捕捉瞬间的快速动作)
        X[f'rel_jit_max{tag}'] = rel_jitter.rolling(w, min_periods=mp).max()
        
        # 3. 目标 B 的平均速度 (B是否不动?)
        b_spd_mean = B_speed.rolling(w, min_periods=mp).mean()
        X[f'target_spd_m{tag}'] = b_spd_mean
        
        # 4. 动静对比 (Agent Jitter / Target Speed)
        # Grooming 典型模式: A Jitter High / B Speed Low
        # 加 epsilon 防止除零
        X[f'groom_ratio_{tag}'] = X[f'rel_jit_m{tag}'] / (b_spd_mean + 0.1)
        
        # 5. 主动方与被动方的震动比 (Active/Passive Jitter Ratio)
        # B 的震动 (如果 B 也在抖，可能是打架)
        B_jitter = np.sqrt(Bx_c.diff().diff()**2 + By_c.diff().diff()**2).fillna(0.0)
        b_jit_mean = B_jitter.rolling(w, min_periods=mp).mean()
        
        X[f'jit_ratio_{tag}'] = X[f'rel_jit_m{tag}'] / (b_jit_mean + 1e-6)

    return X

# %% ==================== 特征工程 pair v206 gpt5.1 lag 改动了内部很多特征 ====================
def add_key_feature_lags_pair(
    X: pd.DataFrame,
    fps: float,
    keys=None,
    lags_base=(5, 10),
    add_future=True,
):
    """
    为少量关键 pair 特征添加显式 lag/lead 列（轻量时序上下文）。
    - lags_base 以 30fps 为基准的“帧数”
    命名:
      {feat}_lag{b}  : feat(t-b)
      {feat}_lead{b} : feat(t+b)
    """
    if keys is None:
        keys = []

    for k in keys:
        if k not in X.columns:
            continue
        for b in lags_base:
            l = _scale(b, fps)
            X[f"{k}_lag{b}"] = X[k].shift(l)
            if add_future:
                X[f"{k}_lead{b}"] = X[k].shift(-l)

    return X

# %% ==================== 特征工程 pair v208 gpt5.1 Egocentric ====================
def _get_part_xy(mouse_pair: pd.DataFrame, side: str, part: str):
    """Safely get x/y Series for a body part."""
    try:
        x = mouse_pair[side][part]['x']
        y = mouse_pair[side][part]['y']
        return x, y
    except Exception:
        return None, None

def _get_side_centroid(mouse_pair: pd.DataFrame, side: str):
    """Centroid of all available keypoints for a mouse side."""
    try:
        parts = mouse_pair[side].columns.get_level_values(0).unique()
    except Exception:
        return None, None

    xs, ys = [], []
    for p in parts:
        x, y = _get_part_xy(mouse_pair, side, p)
        if x is not None:
            xs.append(x)
            ys.append(y)

    if not xs:
        return None, None

    cx = pd.concat(xs, axis=1).mean(axis=1)
    cy = pd.concat(ys, axis=1).mean(axis=1)
    return cx, cy

def _get_side_center(mouse_pair: pd.DataFrame, side: str):
    """Prefer body_center -> neck -> centroid."""
    x, y = _get_part_xy(mouse_pair, side, 'body_center')
    if x is not None:
        return x, y
    x, y = _get_part_xy(mouse_pair, side, 'neck')
    if x is not None:
        return x, y
    return _get_side_centroid(mouse_pair, side)

def _get_heading_vector_A(mouse_pair: pd.DataFrame, avail_A):
    """
    Get agent heading vector (hx, hy) per frame.
    Prefer nose - tail_base, fallback nose - body_center.
    """
    nx, ny = _get_part_xy(mouse_pair, 'A', 'nose')
    if nx is None:
        return None, None

    if 'tail_base' in avail_A:
        tx, ty = _get_part_xy(mouse_pair, 'A', 'tail_base')
        if tx is not None:
            hx = nx - tx
            hy = ny - ty
            return hx, hy

    # fallback: nose - center
    cx, cy = _get_side_center(mouse_pair, 'A')
    if cx is not None:
        hx = nx - cx
        hy = ny - cy
        return hx, hy

    return None, None

def add_within_mouse_elong_pair(X, mouse_pair, avail_A, avail_B):
    """
    Fix the 'elong' idea for pair:
    compute A_elong and B_elong from within-mouse distances.
    """
    def _elong_for(side, avail):
        if not all(p in avail for p in ['nose', 'tail_base', 'ear_left', 'ear_right']):
            return None
        nx, ny = _get_part_xy(mouse_pair, side, 'nose')
        tx, ty = _get_part_xy(mouse_pair, side, 'tail_base')
        elx, ely = _get_part_xy(mouse_pair, side, 'ear_left')
        erx, ery = _get_part_xy(mouse_pair, side, 'ear_right')
        if nx is None or tx is None or elx is None or erx is None:
            return None

        nt = np.sqrt((nx - tx)**2 + (ny - ty)**2)
        ee = np.sqrt((elx - erx)**2 + (ely - ery)**2)
        return nt / (ee + 1e-6)

    A_elong = _elong_for('A', avail_A)
    B_elong = _elong_for('B', avail_B)

    if A_elong is not None:
        X['A_elong'] = A_elong
    if B_elong is not None:
        X['B_elong'] = B_elong
    if A_elong is not None and B_elong is not None:
        X['AB_elong_diff'] = (A_elong - B_elong)
        X['AB_elong_ratio'] = A_elong / (B_elong + 1e-6)

    return X

def add_egocentric_position_features_pair(
    X: pd.DataFrame,
    mouse_pair: pd.DataFrame,
    avail_A,
    avail_B,
    fps: float,
):
    """
    Agent(A)-centric egocentric coordinates for Target(B).
    Outputs signed left/right (x') and front/back (y') in cm.
    """

    # --- origin of A ---
    Acx, Acy = _get_side_center(mouse_pair, 'A')
    if Acx is None:
        return X

    # --- heading of A ---
    hx, hy = _get_heading_vector_A(mouse_pair, avail_A)
    if hx is None:
        return X

    # normalize heading
    h_norm = np.sqrt(hx**2 + hy**2) + 1e-6
    hux = hx / h_norm
    huy = hy / h_norm

    # left-perp unit (x' axis)
    pux = -huy
    puy =  hux

    # --- pick several B reference points ---
    # B center
    Bcx, Bcy = _get_side_center(mouse_pair, 'B')

    # B nose (face proxy)
    Bnx, Bny = _get_part_xy(mouse_pair, 'B', 'nose')

    # B genital proxy: hip_mean if both hips exist else tail_base
    Bhxl, Bhyl = _get_part_xy(mouse_pair, 'B', 'hip_left')
    Bhxr, Bhyr = _get_part_xy(mouse_pair, 'B', 'hip_right')
    if Bhxl is not None and Bhxr is not None:
        Bgx = (Bhxl + Bhxr) / 2.0
        Bgy = (Bhyl + Bhyr) / 2.0
    else:
        Bgx, Bgy = _get_part_xy(mouse_pair, 'B', 'tail_base')

    def _ego_xy(Bx, By, tag):
        if Bx is None:
            return

        rx = Bx - Acx
        ry = By - Acy

        # signed coordinates in A frame
        x_e = (rx * pux + ry * puy)
        y_e = (rx * hux + ry * huy)

        X[f'ego_{tag}_x'] = x_e
        X[f'ego_{tag}_y'] = y_e

        # polar
        X[f'ego_{tag}_r'] = np.sqrt(x_e**2 + y_e**2)
        X[f'ego_{tag}_ang'] = np.arctan2(x_e, y_e)  # left/right signed angle around forward axis

        # quadrant indicators
        X[f'ego_{tag}_front'] = (y_e > 0).astype(float)
        X[f'ego_{tag}_back']  = (y_e < 0).astype(float)
        X[f'ego_{tag}_left']  = (x_e > 0).astype(float)
        X[f'ego_{tag}_right'] = (x_e < 0).astype(float)

        # near-front cone / behind cone (help chase/follow)
        # keep thresholds mild to avoid overfitting
        X[f'ego_{tag}_front_close'] = ((y_e > 0) & (np.abs(x_e) < 5.0)).astype(float)
        X[f'ego_{tag}_back_close']  = ((y_e < 0) & (np.abs(x_e) < 5.0)).astype(float)

        # egocentric velocity of that B point
        vx = x_e.diff().fillna(0.0) * float(fps)
        vy = y_e.diff().fillna(0.0) * float(fps)
        X[f'ego_{tag}_vx'] = vx
        X[f'ego_{tag}_vy'] = vy
        X[f'ego_{tag}_v']  = np.sqrt(vx**2 + vy**2)

        # short rolling stats (fps-adaptive)
        for w_base in [15, 30, 60]:  # 0.5s, 1s, 2s @30fps
            w = _scale(w_base, fps)
            mp = max(1, w // 4)

            X[f'ego_{tag}_xm{w_base}'] = x_e.rolling(w, min_periods=mp).mean()
            X[f'ego_{tag}_xs{w_base}'] = x_e.rolling(w, min_periods=mp).std()
            X[f'ego_{tag}_ym{w_base}'] = y_e.rolling(w, min_periods=mp).mean()
            X[f'ego_{tag}_ys{w_base}'] = y_e.rolling(w, min_periods=mp).std()

            X[f'ego_{tag}_vm{w_base}'] = X[f'ego_{tag}_v'].rolling(w, min_periods=mp).mean()
            X[f'ego_{tag}_vs{w_base}'] = X[f'ego_{tag}_v'].rolling(w, min_periods=mp).std()

    _ego_xy(Bcx, Bcy, "Bc")
    _ego_xy(Bnx, Bny, "Bn")
    _ego_xy(Bgx, Bgy, "Bg")

    # --- optional: relative "who is in front" based on centers ---
    if Bcx is not None:
        # B center projected forward/back relative to A
        # already ego_Bc_y exists if computed
        if 'ego_Bc_y' in X.columns:
            X['ego_B_in_front_rate_1s'] = X['ego_Bc_front'].rolling(_scale(30, fps), min_periods=1).mean()

    return X


# %% ==================== transform ====================
# ---
def transform_pair_v213(
        mouse_pair, 
        body_parts_tracked, 
        fps, 
        pix_per_cm=1.0,
        arena_width_cm=None, 
        arena_height_cm=None, 
        arena_shape=None,
        # ===== NEW =====
        add_key_lags=True,
        key_lags_base=(5, 10, 20, 40),
        key_lag_features=None,
        ):
    # === 空间归一化 (Pixel -> CM) ===
    if abs(pix_per_cm - 1.0) > 1e-4:
        # mouse_pair 包含 'A' 和 'B' 两个一级列
        # 我们需要对整个 DataFrame 除以 scale
        mouse_pair = mouse_pair.copy() / pix_per_cm


    # 获取小鼠A和小鼠B实际存在的身体部位列名
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    # 1. 鼠间距离特征 (Inter-mouse distances)
    # 计算小鼠A的所有部位与小鼠B的所有部位之间的“欧氏距离平方”
    # 使用全排列组合 (Cartesian product)，生成两两部位的距离特征
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    # 重排索引以保证列顺序固定
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    # 2. 类速度特征 (Speed-like features)
    # 计算基于滞后(lag)的位移平方，用于近似速度。Lag根据FPS进行动态缩放（约0.33秒）
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        lag = _scale(10, fps)  # 将10帧(30fps基准)转换为当前fps下的帧数
        shA = mouse_pair['A']['ear_left'].shift(lag)
        shB = mouse_pair['B']['ear_left'].shift(lag)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),  # A的移动量
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False), # A相对于B过去位置的移动量
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False), 
        })
        X = pd.concat([X, speeds], axis=1)

    # 4. 相对朝向 (Relative orientation)
    # 计算小鼠A和B身体轴线（尾部->鼻子）向量的点积，衡量它们是同向、对向还是垂直
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        # 计算余弦相似度 (Cosine Similarity)
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    # 5. 接近速率 (Approach rate)
    # 计算两只老鼠鼻尖距离的变化量。正值表示距离在缩小（正在接近），负值表示远离
    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair['A']['nose'].shift(lag)
        shB_n = mouse_pair['B']['nose'].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    # 6. 距离分箱 (Distance bins)
    # 基于身体中心点的距离，生成离散的布尔特征（单位：cm，不受fps影响）
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt(
            (mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
            (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2
        )
        X['ab_cd'] = cd  # 重要：给 key lag 用

        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls']   = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med']   = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far']   = (cd >= 30.0).astype(float)

        # 原有的中心距离平方时序统计
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)
        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)  # 窗口大小适配FPS
            roll = dict(min_periods=1, center=True)
            # 距离的均值、标准差、最小值、最大值
            X[f'd_m{w}']  = cd_full.rolling(ws, **roll).mean()
            X[f'd_s{w}']  = cd_full.rolling(ws, **roll).std()
            X[f'd_mn{w}'] = cd_full.rolling(ws, **roll).min()
            X[f'd_mx{w}'] = cd_full.rolling(ws, **roll).max()

            # 交互强度倒数 (距离方差越小，数值越大，可能表示稳定的接触或对峙)
            d_var = cd_full.rolling(ws, **roll).var()
            X[f'int{w}'] = 1 / (1 + d_var)

            # 协同运动 (Coordinate movement)
            # 计算A和B在x/y轴位移乘积的和。正值表示同向运动，负值表示反向运动
            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(ws, **roll).mean()
            X[f'co_s{w}'] = coord.rolling(ws, **roll).std()

    # 8. 鼻对鼻动态 (Nose-nose dynamics)
    # 社交行为（如嗅探）的关键特征
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt(
            (mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
            (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2
        )
        X['nn'] = nn  # 重要：给 key lag 用

        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nn_lg{lag}']  = nn.shift(l)        # 滞后的鼻间距
            X[f'nn_ch{lag}']  = nn - nn.shift(l)   # 鼻间距的变化量
            # 过去一段时间内是否处于“近距离接触”状态的比例
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean()

    # 9. 速度对齐度 (Velocity alignment)
    # 计算两只老鼠速度向量的归一化点积（即夹角余弦）。并提取未来/过去的偏移特征
    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        # 速度方向的余弦相似度
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)

        # 提取不同时间偏移(offset)下的对齐度（负数off代表未来，正数代表过去）
        for off in [-20, -10, 0, 10, 20]:
            o = _scale_signed(off, fps)
            X[f'va_{off}'] = val.shift(-o)

        # 距离变异系数
        if 'ab_cd' in X.columns:
            w = _scale(30, fps)
            X['int_con'] = X['ab_cd'].rolling(w, min_periods=1, center=True).std() / \
                           (X['ab_cd'].rolling(w, min_periods=1, center=True).mean() + 1e-6)

        # 10. 高级交互特征 (Advanced interaction)
        # 调用 helper 函数计算追逐(chase)、领头(lead)等复杂逻辑
        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)

    # Sniff proxy features v202
    X = add_sniff_proxy_features_pair(X, mouse_pair, avail_A, avail_B, fps)

    # Pair Grooming Features v203
    X = add_pair_grooming_features(X, mouse_pair, avail_A, avail_B, fps)
    
    # v208 新增内容：
    # --- FIX: within-mouse elong for A/B (optional but recommended) ---
    X = add_within_mouse_elong_pair(X, mouse_pair, avail_A, avail_B)

    # --- NEW: egocentric relative position features ---
    X = add_egocentric_position_features_pair(X, mouse_pair, avail_A, avail_B, fps)

    # 关键特征统一 lag/lead v206
    if add_key_lags:
        if key_lag_features is None:
            key_lag_features = [
                "ab_cd",
                "nn",
                "appr",
                "int_con",
                "chase_30",
                "snf_dmin_face",
                "snf_dmin_genital",
                "snf_d_body",
                "A_elong",
                "B_elong",
            ]
        X = add_key_feature_lags_pair(
            X,
            fps=fps,
            keys=key_lag_features,
            lags_base=key_lags_base,
            add_future=True,
        )

    return X.astype(np.float32, copy=False)
# ---