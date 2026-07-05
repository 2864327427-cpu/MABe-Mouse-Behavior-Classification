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

def add_key_feature_lags(
    X: pd.DataFrame,
    fps: float,
    keys=None,
    lags_base=(1, 5),
    add_future=True,
):
    """
    为少量关键特征添加显式 lag/lead 列（轻量时序上下文）。
    - lags_base 以 30fps 为基准的“帧数”
    - 会用 _scale 自动按 fps 缩放
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
            # past
            X[f"{k}_lag{b}"] = X[k].shift(l)
            # future (non-causal)
            if add_future:
                X[f"{k}_lead{b}"] = X[k].shift(-l)

    return X

def _speed(cx: pd.Series, cy: pd.Series, fps: float) -> pd.Series:
    """计算瞬时速度 (cm/s)。使用 hypot 计算欧几里得位移，并乘以 FPS 归一化到秒。"""
    return np.hypot(cx.diff(), cy.diff()).fillna(0.0) * float(fps)

def _roll_future_mean(s: pd.Series, w: int, min_p: int = 1) -> pd.Series:
    """
    计算“未来”时间窗口的均值（非因果特征）。
    实现方式：先反转序列，做常规 rolling，再反转回来。
    区间: [t, t+w-1]
    """
    return s.iloc[::-1].rolling(w, min_periods=min_p).mean().iloc[::-1]

def _roll_future_var(s: pd.Series, w: int, min_p: int = 2) -> pd.Series:
    """
    计算“未来”时间窗口的方差。
    区间: [t, t+w-1]
    """
    return s.iloc[::-1].rolling(w, min_periods=min_p).var().iloc[::-1]

def add_curvature_features(X, center_x, center_y, fps):
    """
    计算轨迹曲率特征 (Trajectory curvature)。
    曲率反映了运动路径的弯曲程度（如急转弯 vs 直线跑）。
    """
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    # 计算二维曲率公式: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)  # 增加 epsilon 防止除零

    for w in [15, 30, 60]:
        ws = _scale(w, fps)
        # 计算平滑后的平均曲率
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    # 计算转向率 (Turn Rate)：速度向量角度的变化量
    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    for w in [15, 30]:
        ws = _scale(w, fps)
        X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    """
    多尺度时间特征 (Multi-scale temporal features)。
    计算不同时间窗口下的速度均值和标准差，用于区分短时爆发运动和长时持续运动。
    """
    # 此时 displacement 已经是 cm 单位，乘以 FPS 转换为 cm/s
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [5, 10, 20, 40, 160, 320]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    # 短/长速度比值
    if f'sp_m5' in X.columns and f'sp_m40' in X.columns:
        X['sp_ratio_5_40'] = X['sp_m5'] / (X['sp_m40'] + 1e-6)
    if f'sp_m10' in X.columns and f'sp_m160' in X.columns:
        X['sp_ratio_10_160'] = X['sp_m10'] / (X['sp_m160'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y, fps):
    """
    行为状态转换特征 (Behavioral state transitions) - FIXED
    修复了 bins 错误乘以 fps 的 bug。
    """
    # 计算瞬时速度 (cm/s)
    # diff() 计算的是帧间距离，乘以 fps 转换为每秒距离
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    
    # 平滑窗口：0.5秒 (15帧 @ 30fps)
    w_ma = max(1, _scale(15, fps))
    
    # 平滑处理，减少噪声导致的频繁状态跳变
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        # === 修复 Bug ===
        # bins 应该是物理速度阈值 (cm/s)，不应再乘以 fps
        # 状态: 0=静止(<0.5), 1=微动(0.5-2.0), 2=移动(2.0-5.0), 3=快跑(>5.0)
        bins = [-np.inf, 0.5, 2.0, 5.0, np.inf]
        
        # 使用 pd.cut 进行离散化
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [30, 60, 120]: # 基准 1秒, 2秒, 4秒
            ws = _scale(window, fps)
            # 确保数据长度足够
            if len(speed_states) >= ws:
                min_p = max(1, ws // 6)
                
                # 1. 计算窗口内各状态的占比 (Duration percentage)
                # 显式遍历状态 0, 1, 2, 3
                for state in [0, 1, 2, 3]:
                    # 使用 1.0/0.0 boolean mask 的 rolling mean 即为占比
                    mask = (speed_states == state).astype(float)
                    X[f's{state}_{window}'] = mask.rolling(ws, min_periods=min_p).mean()

                # 2. 计算状态切换次数 (State transitions count)
                # 反映行为的碎片化程度 (Fragmentation)
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                # 第一帧通常是 NaN 或 1，这里 fillna(0) 比较稳妥
                state_changes = state_changes.fillna(0.0)
                
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=min_p).sum()
                
    except Exception as e:
        # 生产环境中建议 print(e) 以便调试，或者 pass 保证流程不中断
        # print(f"Error in add_state_features: {e}")
        pass

    return X

def add_longrange_features(X, center_x, center_y, fps):
    """
    长时程时序特征 (Long-range temporal features)。
    使用大窗口计算位置和速度的趋势，捕捉如“巡逻”、“定居”等宏观模式。
    """
    for window in [120, 240]: # 基准 4秒, 8秒
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            # 长窗口位置均值
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    # 使用指数加权移动平均 (EWM) 提取特征，span 参数也需随 FPS 缩放
    for span in [60, 120, 240]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in [60, 120, 240]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            # 速度在窗口内的百分位排名 (Percentile rank)，判断当前是处于该时段的相对高峰还是低谷
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X

def add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base: int = 180, colname: str = "path_cum180"):
    """
    计算长窗口内的累积路径长度 (Cumulative Path Length)。
    horizon_frames_base: 基准窗口长度（帧）
    """
    L = max(1, _scale(horizon_frames_base, fps))  # 转换窗口
    # 单帧步长 (cm)
    step = np.hypot(cx.diff(), cy.diff())
    # 计算中心化窗口内的总路程 (双边求和 ~2L+1 帧)，反映该时间段的总运动量
    path = step.rolling(2*L + 1, min_periods=max(5, L//6), center=True).sum()
    X[colname] = path.fillna(0.0).astype(np.float32)
    return X

def add_groom_microfeatures(X, df, fps):
    """
    Grooming 专用多尺度微观特征（新命名、不兼容旧字段）。

    直觉：
    - Grooming 常见模式 = 身体（中心）相对稳定 + 头部/鼻子高频小幅运动
    - 因此用 “鼻速 / 身体速” 的短窗统计 + 鼻子相对身体的径向抖动
    - 再加可选的头部朝向抖动（需要 tail_base）

    多尺度窗口（以 30fps 基准帧数）：
    - 6  -> ~0.2s
    - 9  -> ~0.3s
    - 15 -> ~0.5s
    - 30 -> ~1.0s
    _scale 会自动适配不同 fps
    """
    parts = df.columns.get_level_values(0)

    # --- 必需：nose ---
    if 'nose' not in parts:
        return X

    # --- 选择“身体参考点”(center) ---
    if 'body_center' in parts:
        cx = df['body_center']['x']; cy = df['body_center']['y']
    elif 'neck' in parts:
        cx = df['neck']['x']; cy = df['neck']['y']
    else:
        # fallback: 全部可用关键点的质心
        xs = df.xs('x', level=1, axis=1)
        ys = df.xs('y', level=1, axis=1)
        cx = xs.mean(axis=1)
        cy = ys.mean(axis=1)

    nx = df['nose']['x']; ny = df['nose']['y']

    # --- 速度（单位依赖你上游坐标尺度；若已做 cm/arena 归一会更稳）---
    cs = (np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)).fillna(0.0)
    ns = (np.sqrt(nx.diff()**2 + ny.diff()**2) * float(fps)).fillna(0.0)

    # --- 头身解耦比值 ---
    eps = 1e-3
    ratio = (ns / (cs + eps)).clip(0, 10)

    # --- 鼻子相对身体中心的径向距离 ---
    r = np.sqrt((nx - cx)**2 + (ny - cy)**2)

    # --- 头部朝向抖动（可选）---
    has_tail = 'tail_base' in parts
    if has_tail:
        ang = np.arctan2(
            df['nose']['y'] - df['tail_base']['y'],
            df['nose']['x'] - df['tail_base']['x']
        )
        dang = np.abs(ang.diff()).fillna(0.0)

    base_windows = [6, 9, 15, 30]  # frames @30fps

    def _tag_from_base_frames(bw):
        sec = bw / 30.0
        # 0.2 -> "0p2s"
        return f"{sec:.1f}".replace('.', 'p') + "s"

    for bw in base_windows:
        w = _scale(bw, fps)
        mp = max(1, w // 3)
        tag = _tag_from_base_frames(bw)

        # 1) 头身解耦：rolling median/mean（高频 + 稳态两种视角）
        X[f'groom_hbd_med_{tag}']  = ratio.rolling(w, min_periods=mp).median()
        X[f'groom_hbd_mean_{tag}'] = ratio.rolling(w, min_periods=mp).mean()

        # 2) 鼻子径向抖动
        X[f'groom_nose_rad_std_{tag}'] = r.rolling(w, min_periods=mp).std().fillna(0.0)

        # 3) 身体/鼻子的独立强度（给模型更多可分解信号）
        X[f'groom_cs_mean_{tag}'] = cs.rolling(w, min_periods=mp).mean()
        X[f'groom_ns_mean_{tag}'] = ns.rolling(w, min_periods=mp).mean()

        # 4) 头部朝向抖动
        if has_tail:
            X[f'groom_orient_jit_mean_{tag}'] = dang.rolling(w, min_periods=mp).mean()
            X[f'groom_orient_jit_std_{tag}']  = dang.rolling(w, min_periods=mp).std().fillna(0.0)

    return X

def add_speed_asymmetry_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    horizon_base: int = 30, agg: str = "mean"
) -> pd.DataFrame:
    '''
    # 过去 vs 未来 速度不对称性 (Past–vs–Future speed asymmetry)
    # 计算 Δv = 未来均速 - 过去均速
    # 非因果特征，用于检测动作的起始（突然加速）或结束（突然停止）
    '''
    w = max(3, _scale(horizon_base, fps))
    v = _speed(cx, cy, fps)
    if agg == "median":
        # 过去 w 帧的中位数
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).median()
        # 未来 w 帧的中位数
        v_fut  = v.iloc[::-1].rolling(w, min_periods=max(3, w//4)).median().iloc[::-1]
    else:
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).mean()
        v_fut  = _roll_future_mean(v, w, min_p=max(3, w//4))
    
    X["spd_asym_1s"] = (v_fut - v_past).fillna(0.0)
    return X


def add_gauss_shift_speed_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    window_base: int = 30, eps: float = 1e-6
) -> pd.DataFrame:
    '''
    # 速度分布漂移 (Distribution shift via Symmetric KL divergence)
    # 假设过去和未来的速度分布都服从高斯分布 N(μ, σ²)。
    # 计算对称 KL 散度来衡量这两个分布的差异。
    # 数值越大，说明当前时刻前后运动模式发生了剧烈改变（例如从静止突然爆发）。
    '''
    w = max(5, _scale(window_base, fps))
    v = _speed(cx, cy, fps)

    # 过去的均值与方差
    mu_p = v.rolling(w, min_periods=max(3, w//4)).mean()
    va_p = v.rolling(w, min_periods=max(3, w//4)).var().clip(lower=eps)

    # 未来的均值与方差
    mu_f = _roll_future_mean(v, w, min_p=max(3, w//4))
    va_f = _roll_future_var(v, w, min_p=max(3, w//4)).clip(lower=eps)

    # 计算对称 KL 散度: KL(Past||Future) + KL(Future||Past)
    # 高斯分布间的 KL 散度有解析解，计算效率高
    kl_pf = 0.5 * ((va_p/va_f) + ((mu_f - mu_p)**2)/va_f - 1.0 + np.log(va_f/va_p))
    kl_fp = 0.5 * ((va_f/va_p) + ((mu_p - mu_f)**2)/va_p - 1.0 + np.log(va_p/va_f))
    X["spd_symkl_1s"] = (kl_pf + kl_fp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# %% ==================== 特征工程V2 061759 ====================
def add_morphometric_pca(X, single_mouse, body_parts_tracked, fps):
    """
    瞬时形态学PCA特征 (Per-frame PCA)
    不依赖具体的部位名称，而是将老鼠看作一个点云。
    计算每一帧点云的协方差矩阵的特征值，提取身体的主轴长度(L1)和次轴宽度(L2)。
    
    物理意义：
    - L1 (Major Axis): 身体的延展长度。
    - L2 (Minor Axis): 身体的宽度/肥胖度。
    - Eccentricity (离心率): L1/L2，衡量老鼠是像个"球"(Sleeping)还是像根"棍"(Running)。
    """
    # 提取所有存在的x和y坐标
    xs = single_mouse.xs('x', level=1, axis=1)
    ys = single_mouse.xs('y', level=1, axis=1)
    
    # 中心化
    mean_x = xs.mean(axis=1)
    mean_y = ys.mean(axis=1)
    xs_c = xs.sub(mean_x, axis=0)
    ys_c = ys.sub(mean_y, axis=0)
    
    # 向量化计算2x2协方差矩阵 entries: Var(x), Var(y), Cov(x,y)
    # Cov = (X.T @ X) / (N-1)
    N = xs.shape[1]
    var_x = (xs_c ** 2).sum(axis=1) / (N - 1)
    var_y = (ys_c ** 2).sum(axis=1) / (N - 1)
    cov_xy = (xs_c * ys_c).sum(axis=1) / (N - 1)
    
    # 2x2 矩阵特征值解析解:
    # lambda = ((a+d) +/- sqrt((a-d)^2 + 4b^2)) / 2
    # 其中 a=var_x, d=var_y, b=cov_xy
    trace = var_x + var_y
    det = var_x * var_y - cov_xy ** 2
    # 判别式 delta
    delta = np.sqrt((var_x - var_y)**2 + 4 * cov_xy**2)
    
    eig1 = (trace + delta) / 2.0
    eig2 = (trace - delta) / 2.0
    
    # 特征工程
    # 主轴长度 (近似)
    X['morph_len'] = np.sqrt(eig1)
    # 次轴长度 (近似宽度)
    X['morph_wid'] = np.sqrt(eig2)
    # 面积 (近似)
    X['morph_area'] = X['morph_len'] * X['morph_wid']
    # 离心率 (Aspect Ratio): 越大越细长，越接近1越圆
    X['morph_ratio'] = X['morph_len'] / (X['morph_wid'] + 1e-6)
    
    # 加入时序平滑
    w = _scale(15, fps)
    X['morph_ratio_ma'] = X['morph_ratio'].rolling(w, min_periods=1).mean()
    
    return X

def add_body_bend_angle(X, single_mouse, avail_parts):
    """
    脊柱弯曲角度 (Spine Bend / Curl)
    计算 <鼻子-中心> 向量与 <中心-尾基> 向量的夹角。
    
    物理意义：
    - 接近 180度 (pi): 老鼠身体笔直 (Running/Stretching)。
    - 角度急剧减小: 老鼠身体弯曲 (Turning/Grooming tail)。
    """
    if all(p in avail_parts for p in ['nose', 'body_center', 'tail_base']):
        # Vector 1: Center -> Nose
        v1x = single_mouse['nose']['x'] - single_mouse['body_center']['x']
        v1y = single_mouse['nose']['y'] - single_mouse['body_center']['y']
        
        # Vector 2: Center -> Tail_base (注意方向，为了算夹角通常取从中心出发)
        v2x = single_mouse['tail_base']['x'] - single_mouse['body_center']['x']
        v2y = single_mouse['tail_base']['y'] - single_mouse['body_center']['y']
        
        # Dot product & Magnitudes
        dot = v1x * v2x + v1y * v2y
        mag1 = np.sqrt(v1x**2 + v1y**2)
        mag2 = np.sqrt(v2x**2 + v2y**2)
        
        # Cosine angle
        cos_angle = dot / (mag1 * mag2 + 1e-6)
        # 裁剪到 [-1, 1] 防止数值误差导致 arccos 报错
        cos_angle = cos_angle.clip(-1.0, 1.0)
        
        # 角度 (弧度) -> 转换为度数更容易理解特征
        angle = np.arccos(cos_angle) * (180.0 / np.pi)
        
        X['spine_angle'] = angle.fillna(180.0) # 默认为直
        
        # 脊柱弯曲的变异性 (Grooming时会扭动)
        X['spine_var_30'] = X['spine_angle'].rolling(30, min_periods=1).std()
        
    return X

def add_tortuosity_efficiency(X, cx, cy, fps):
    """
    路径曲折度与运动效率 (Tortuosity & Efficiency)
    
    物理意义：
    - Net Displacement (直线距离) vs Path Distance (实际路程)
    - Efficiency = Net / Path。
    - 接近 1.0: 有目的的移动 (Transit)。
    - 接近 0.0: 原地打转或徘徊 (Local exploration / Grooming)。
    """
    for w in [30, 90]: # 1秒, 3秒
        ws = _scale(w, fps)
        
        # 1. Net Displacement (起点到终点的直线距离)
        dx_net = cx.diff(ws)
        dy_net = cy.diff(ws)
        net_disp = np.sqrt(dx_net**2 + dy_net**2)
        
        # 2. Path Distance (窗口内所有步长之和)
        step_len = np.sqrt(cx.diff()**2 + cy.diff()**2)
        path_dist = step_len.rolling(ws, min_periods=1).sum()
        
        # 3. Efficiency Index (Tortuosity的倒数)
        # 加 epsilon 防止除零，静止时 efficiency 定义为 0
        X[f'path_eff_{w}'] = net_disp / (path_dist + 1e-6)
        
        # 4. Fractal Dimension approximation (分形维数近似)
        # log(Path) / log(Net) 也是一种常用的生物学特征
        # X[f'fractal_{w}'] = np.log(path_dist + 1e-6) / (np.log(net_disp + 1e-6) + 1e-6)
        
    return X

def add_frequency_jitter(X, single_mouse, avail_parts, fps):
    """
    高频微颤特征 (High-Frequency Jitter)
    Grooming (梳理) 动作通常包含高频、低幅度的身体震动，单纯的速度特征很难捕捉。
    这里使用高通滤波后的能量来表示。
    """
    target_parts = ['nose', 'head', 'ear_left'] # 头部震动最明显
    
    for part in target_parts:
        # 兼容不同命名 (head 有时不存在)
        if part not in avail_parts: continue
            
        px = single_mouse[part]['x']
        py = single_mouse[part]['y']
        
        # 计算瞬时加加速度 (Jerk) 或 简单的高频差分
        # 这里使用 diff().diff() 近似高频分量 (二阶差分)
        acc_mag = np.sqrt(px.diff().diff()**2 + py.diff().diff()**2)
        
        for base in [6, 9, 15]:
            w = _scale(base, fps)
            # 计算高频能量均值
            X[f'{part}_jitter_mean_{base}'] = acc_mag.rolling(w, min_periods=1).mean()
            # 计算高频能量的最大值 (捕捉爆发性抽动)
            X[f'{part}_jitter_max_{base}']  = acc_mag.rolling(w, min_periods=1).max()

    return X


# %% ==================== 特征工程V5 ====================
def add_triangle_areas_single(X, df, body_parts_tracked):
    """
    计算关键点构成的三角形面积特征。
    物理意义：顶视视角下，当老鼠 Rear（站立）或 Huddle（蜷缩）时，特定的三角形面积会显著缩小或发生形变。
    """
    # 1. 头部三角形 (Head Triangle): 鼻 - 左耳 - 右耳
    # 这对于检测 Rear 非常敏感，因为站立时鼻子会后缩，导致投影面积变化
    if all(p in body_parts_tracked for p in ['nose', 'ear_left', 'ear_right']):
        # 向量叉乘公式求面积: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        x1, y1 = df['nose']['x'], df['nose']['y']
        x2, y2 = df['ear_left']['x'], df['ear_left']['y']
        x3, y3 = df['ear_right']['x'], df['ear_right']['y']
        
        area = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        X['area_head'] = area
        
        # 面积的变化率 (捕捉突然的站起/落下)
        X['area_head_ch'] = area.diff().fillna(0)

    # 2. 身体主三角形 (Body Main Triangle): 鼻 - 尾根 - 身体中心 (如果存在) 或 颈部
    # 反映身体的弯曲程度 (Curvature/Bending)
    targets = ['nose', 'tail_base']
    third_pt = 'body_center' if 'body_center' in body_parts_tracked else ('neck' if 'neck' in body_parts_tracked else None)
    
    if third_pt and all(p in body_parts_tracked for p in targets):
        x1, y1 = df['nose']['x'], df['nose']['y']
        x2, y2 = df['tail_base']['x'], df['tail_base']['y']
        x3, y3 = df[third_pt]['x'], df[third_pt]['y']
        
        area_body = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        X['area_body'] = area_body
        
        # 形状因子：面积 / (鼻尾距离平方)。消除个体大小差异，纯粹反映姿态
        nt_dist_sq = (x1-x2)**2 + (y1-y2)**2
        X['shape_factor'] = area_body / (nt_dist_sq + 1e-6)

    return X

def add_body_part_dissociation(X, df, fps, body_parts_tracked):
    """
    计算身体部位运动的解耦特征 (Dissociation)。
    物理意义：Dig(挖) 和 Groom(梳) 的典型特征是身体某一部分剧烈运动，而另一部分相对静止。
    """
    # 需要的前部部位 (Front) 和 后部部位 (Rear)
    front_parts = [p for p in ['nose', 'ear_left', 'ear_right'] if p in body_parts_tracked]
    rear_parts = [p for p in ['tail_base', 'hip_left', 'hip_right'] if p in body_parts_tracked]
    
    if not front_parts or not rear_parts:
        return X

    # 计算前部平均速度 (标量)
    v_front = pd.Series(0.0, index=df.index)
    for p in front_parts:
        v = np.sqrt(df[p]['x'].diff()**2 + df[p]['y'].diff()**2)
        v_front += v
    v_front /= len(front_parts)
    v_front = v_front.fillna(0) * float(fps)

    # 计算后部平均速度 (标量)
    v_rear = pd.Series(0.0, index=df.index)
    for p in rear_parts:
        v = np.sqrt(df[p]['x'].diff()**2 + df[p]['y'].diff()**2)
        v_rear += v
    v_rear /= len(rear_parts)
    v_rear = v_rear.fillna(0) * float(fps)
    
    w = int(0.5 * fps) # 0.5秒窗口

    # 特征1: 前后动能比 (Front-Rear Ratio)
    # Dig/Groom时，front 很大，rear 很小 -> ratio 高
    # Run 时，front 和 rear 都很大 -> ratio 趋近 1
    # Rest 时，都接近 0 -> ratio 不稳定 (使用 log 处理或加 epsilon)
    X['fr_ratio'] = np.log1p(v_front) - np.log1p(v_rear)
    X['fr_ratio_ma'] = X['fr_ratio'].rolling(w, min_periods=1).mean()

    # 特征2: 局部震颤 (Local Jitter / High Frequency Energy)
    # 计算加速度变化的绝对值，反映“抖动”
    acc_front = v_front.diff().abs()
    X['front_jitter'] = acc_front.rolling(w, min_periods=1).mean()
    
    return X

def add_tortuosity_features(X, cx, cy, fps):
    """
    计算路径曲折度 (Tortuosity / Straightness Index)。
    物理意义：区分 Run (直线，低曲折度) 和 Explore/Sniff (蜿蜒，高曲折度)。
    公式：路径总长度 / 起终点欧氏距离
    """
    # 窗口：1秒 和 3秒
    for win_sec in [1.0, 3.0]:
        w = max(5, int(win_sec * fps))
        
        # 1. 窗口内的路径总长度 (Path Length)
        step_dist = np.sqrt(cx.diff()**2 + cy.diff()**2).fillna(0)
        path_len = step_dist.rolling(w, min_periods=w//2).sum()
        
        # 2. 窗口起终点的直线距离 (Displacement)
        # diff(w) 计算的是 index i 和 index i-w 的坐标差
        disp_dist = np.sqrt(cx.diff(w)**2 + cy.diff(w)**2).fillna(0)
        
        # 3. 曲折度 (Tortuosity)
        # 值越接近 1 表示越直，值越大表示越蜿蜒/原地打转
        # 加上 epsilon 防止除零 (静止时 disp_dist 为 0)
        X[f'tortuosity_{int(win_sec)}s'] = path_len / (disp_dist + 0.1)
        
        # 4. 净位移效率 (Efficiency)
        # 反向特征：位移 / (路程 + 1)
        X[f'move_eff_{int(win_sec)}s'] = disp_dist / (path_len + 1.0)

    return X

# %% ==================== 特征工程 v51 ====================
def add_center_kinematics_v2(X, cx, cy, fps):
    """
    ===== NEW =====
    身体/质心的多阶运动学时序特征：
    - 多lag位移/平均速度
    - 速度/加速度/jerk 的多窗口统计
    - 短-长窗口对比(contrast)
    
    这组特征重点增强：
    1) 动作起始/结束的“瞬态变化”
    2) rest/freeze vs explore/run 的“强弱与稳定性”
    """
    v = _speed(cx, cy, fps)  # cm/s

    # 1) 多lag位移 与 平均速度
    # 基于30fps的帧尺度
    for lag_base in [1, 2, 5, 10, 20, 40]:
        l = _scale(lag_base, fps)
        dx = cx - cx.shift(l)
        dy = cy - cy.shift(l)
        disp = np.hypot(dx, dy).fillna(0.0)
        X[f'c_disp_l{lag_base}'] = disp
        # 平均速度（该lag跨度内）
        X[f'c_sp_l{lag_base}'] = disp * float(fps) / max(1, l)

    # 2) 加速度/jerk（用一阶/二阶差分近似）
    # 标量加速度 ~ dv/dt
    a = (v.diff().fillna(0.0) * float(fps))
    j = (a.diff().fillna(0.0) * float(fps))

    X['c_v'] = v
    X['c_a'] = a
    X['c_j'] = j

    # 3) 多窗口统计（短/中/长）
    for w_base in [5, 15, 30, 60, 120]:
        w = _scale(w_base, fps)
        mp = max(1, w // 4)

        X[f'c_v_m{w_base}'] = v.rolling(w, min_periods=mp).mean()
        X[f'c_v_s{w_base}'] = v.rolling(w, min_periods=mp).std()
        X[f'c_v_max{w_base}'] = v.rolling(w, min_periods=mp).max()

        X[f'c_a_m{w_base}'] = a.rolling(w, min_periods=mp).mean()
        X[f'c_a_s{w_base}'] = a.rolling(w, min_periods=mp).std()

        X[f'c_j_m{w_base}'] = j.rolling(w, min_periods=mp).mean()

        # 速度变异系数：稳定移动 vs 抖动碎片化
        X[f'c_v_cv{w_base}'] = X[f'c_v_s{w_base}'] / (X[f'c_v_m{w_base}'] + 1e-6)

    # 4) 短-长对比（强调状态切换）
    w_s = _scale(15, fps)
    w_l = _scale(90, fps)
    ms = max(1, w_s // 3)
    ml = max(1, w_l // 3)

    v_s = v.rolling(w_s, min_periods=ms).mean()
    v_l = v.rolling(w_l, min_periods=ml).mean()

    X['c_v_contrast_15_90'] = (v_s - v_l).fillna(0.0)
    X['c_v_ratio_15_90'] = (v_s / (v_l + 1e-6)).fillna(0.0)

    # 5) 事件性 / burstiness：窗口内“超阈值活跃比例”
    # 阈值用相对稳健方式：短窗均速的分位数代理（简化为 rolling median + 常数）
    v_med = v.rolling(_scale(60, fps), min_periods=1).median()
    active = (v > (v_med + 1.0)).astype(float)  # +1cm/s 作为小幅偏置
    X['c_active_1s'] = active.rolling(_scale(30, fps), min_periods=1).mean()
    X['c_active_3s'] = active.rolling(_scale(90, fps), min_periods=1).mean()

    return X

def add_part_micro_motion_v2(X, single_mouse, avail_parts, fps):
    """
    ===== NEW =====
    关键部位的“局部时序微运动”：
    - nose / ears / tail_base（若存在）
    - 多lag 位移、rolling 抖动强度
    
    让 selfgroom / exploreobject / rear / rest 的区分更稳。
    """
    key_parts = [p for p in ['nose', 'ear_left', 'ear_right', 'tail_base'] if p in avail_parts]
    if not key_parts:
        return X

    for p in key_parts:
        px = single_mouse[p]['x']
        py = single_mouse[p]['y']
        # 局部瞬时速度
        pv = np.hypot(px.diff(), py.diff()).fillna(0.0) * float(fps)

        X[f'{p}_v'] = pv

        # 多lag局部位移/平均速度
        for lag_base in [2, 5, 10, 20]:
            l = _scale(lag_base, fps)
            disp = np.hypot(px - px.shift(l), py - py.shift(l)).fillna(0.0)
            X[f'{p}_disp_l{lag_base}'] = disp
            X[f'{p}_sp_l{lag_base}'] = disp * float(fps) / max(1, l)

        # rolling 抖动强度（短/中）
        for w_base in [10, 30, 60]:
            w = _scale(w_base, fps)
            mp = max(1, w // 4)
            X[f'{p}_v_m{w_base}'] = pv.rolling(w, min_periods=mp).mean()
            X[f'{p}_v_s{w_base}'] = pv.rolling(w, min_periods=mp).std()

    # 左右耳对称性：对 rear/turn/梳理的辅助判别
    if 'ear_left' in avail_parts and 'ear_right' in avail_parts:
        elv = X.get('ear_left_v', None)
        erv = X.get('ear_right_v', None)
        if elv is not None and erv is not None:
            X['ear_lr_v_diff'] = (elv - erv).abs()  # ===== NEW =====
            X['ear_lr_v_ratio'] = elv / (erv + 1e-6)  # ===== NEW =====

    return X

def add_nt_dynamic_v2(X, single_mouse, avail_parts, fps):
    """
    ===== NEW =====
    鼻-尾基距离的“更密集时序刻画”：
    - 多lag shift
    - rolling 均值/方差/变化率
    
    强化：
    - rear（伸展/站立）
    - selfgroom（蜷/折）
    - huddle/rest（收缩稳定）
    """
    if not all(p in avail_parts for p in ['nose', 'tail_base']):
        return X

    nt = np.sqrt(
        (single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
        (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2
    )

    X['nt_dist'] = nt

    # 多lag
    for lag_base in [5, 10, 20, 40, 80]:
        l = _scale(lag_base, fps)
        X[f'nt_lg{lag_base}'] = nt.shift(l)
        X[f'nt_df{lag_base}'] = (nt - nt.shift(l)).fillna(0.0)

    # rolling 统计
    for w_base in [15, 30, 60, 120]:
        w = _scale(w_base, fps)
        mp = max(1, w // 4)
        X[f'nt_m{w_base}'] = nt.rolling(w, min_periods=mp).mean()
        X[f'nt_s{w_base}'] = nt.rolling(w, min_periods=mp).std()
        X[f'nt_cv{w_base}'] = X[f'nt_s{w_base}'] / (X[f'nt_m{w_base}'] + 1e-6)

    # 形态变化“速度”
    nt_v = (nt.diff().fillna(0.0) * float(fps)).abs()
    X['nt_v'] = nt_v
    X['nt_v_m30'] = nt_v.rolling(_scale(30, fps), min_periods=1).mean()

    return X


# %% ==================== 特征工程 v52 ====================
def add_arena_spatial_features_single(
    X: pd.DataFrame,
    cx: pd.Series,
    cy: pd.Series,
    fps: float,
    arena_w_cm: float,
    arena_h_cm: float,
    arena_shape: str = "square",
):
    if arena_w_cm is None or arena_h_cm is None:
        return X
    aw = float(arena_w_cm)
    ah = float(arena_h_cm)
    if aw <= 0 or ah <= 0:
        return X

    shape = str(arena_shape).lower() if arena_shape is not None else "square"

    # 归一化坐标（鲁棒）
    X["arena_x_norm"] = (cx / (aw + 1e-6)).clip(0, 1)
    X["arena_y_norm"] = (cy / (ah + 1e-6)).clip(0, 1)

    # 到中心距离
    dx_c = cx - aw / 2.0
    dy_c = cy - ah / 2.0
    center_dist = np.sqrt(dx_c**2 + dy_c**2)
    X["arena_center_dist"] = center_dist

    if "circ" in shape:
        # 近似圆形场地
        r = min(aw, ah) / 2.0
        wall_dist = (r - center_dist).clip(lower=0)
        X["arena_wall_dist"] = wall_dist
        # 圆形没有明确“角”
        X["arena_corner_dist"] = wall_dist  # 作为弱替代
    else:
        # 矩形/方形/分割矩形：按矩形处理
        d_left = cx
        d_right = aw - cx
        d_top = cy
        d_bottom = ah - cy

        # 最近墙距离（负值说明坐标/尺度不一致，clip 掉）
        wall_dist = pd.concat([d_left, d_right, d_top, d_bottom], axis=1).min(axis=1)
        wall_dist = wall_dist.clip(lower=0)
        X["arena_wall_dist"] = wall_dist

        # 最近角距离
        # 四角
        c1 = np.sqrt((cx - 0.0)**2 + (cy - 0.0)**2)
        c2 = np.sqrt((cx - aw)**2 + (cy - 0.0)**2)
        c3 = np.sqrt((cx - 0.0)**2 + (cy - ah)**2)
        c4 = np.sqrt((cx - aw)**2 + (cy - ah)**2)
        corner_min = pd.concat([c1, c2, c3, c4], axis=1).min(axis=1)
        X["arena_corner_dist"] = corner_min

    # 近墙/近角指示 + 1 秒占比（更鲁棒）
    # 阈值用 cm，偏保守
    wall_close = (X["arena_wall_dist"] < 2.0).astype(float)
    corner_close = (X["arena_corner_dist"] < 3.0).astype(float)

    X["arena_wall_close_2cm"] = wall_close
    X["arena_corner_close_3cm"] = corner_close

    w1s = _scale(30, fps)  # 1s @30fps 基准
    mp = max(1, w1s // 3)
    X["arena_wall_close_rate_1s"] = wall_close.rolling(w1s, min_periods=mp).mean()
    X["arena_corner_close_rate_1s"] = corner_close.rolling(w1s, min_periods=mp).mean()

    return X



# %% ==================== transform ====================
# from gpt5.1
def transform_single_v52(
        single_mouse, 
        body_parts_tracked, 
        fps, 
        pix_per_cm=1.0, 
        arena_width_cm=None,
        arena_height_cm=None,
        arena_shape=None,
        add_key_lags=True,
        key_lags_base=(1, 5),
        key_lag_features=None,
        ):
    # 只要在这里处理了，后续所有基于坐标的特征（速度、距离、面积）都会自动变成 cm 单位
    if abs(pix_per_cm - 1.0) > 1e-4: 
        # 使用 xs 选取并修改，再赋值回去可能会丢失层级，
        # 最稳妥的方法是直接除整个 DataFrame 的数值，因为输入只有 x, y 列
        # 但要小心 single_mouse 可能包含 bodypart 索引
        # 简单暴力的做法：直接除。因为输入通常只包含数值列。
        single_mouse = single_mouse.copy() / pix_per_cm


    available_body_parts = single_mouse.columns.get_level_values(0)

    # --- 1. 基础距离特征 ---
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(
        columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)],
        copy=False
    )

    # --- 2. 速度类特征（耳/尾） ---
    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        speeds = pd.DataFrame({
            'sp_lf':  np.square(single_mouse['ear_left']  - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt':  np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left']  - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    # --- 3. 身体伸长率 ---
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # --- 4. 选择中心点 cx/cy ---
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
    else:
        # Fallback: centroid
        cx = single_mouse.xs('x', level=1, axis=1).mean(axis=1)
        cy = single_mouse.xs('y', level=1, axis=1).mean(axis=1)

    # --- 5. 身体角度/弯曲度 ---
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    # --- 6. 核心时序统计特征（中心轨迹） ---
    for w in [5, 15, 30, 60]:
        ws = _scale(w, fps)
        roll = dict(min_periods=1, center=True)

        X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
        X[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
        X[f'cx_s{w}'] = cx.rolling(ws, **roll).std()
        X[f'cy_s{w}'] = cy.rolling(ws, **roll).std()
        X[f'x_rng{w}'] = cx.rolling(ws, **roll).max() - cx.rolling(ws, **roll).min()
        X[f'y_rng{w}'] = cy.rolling(ws, **roll).max() - cy.rolling(ws, **roll).min()
        X[f'disp{w}'] = np.sqrt(
            cx.diff().rolling(ws, min_periods=1).sum()**2 +
            cy.diff().rolling(ws, min_periods=1).sum()**2
        )
        X[f'act{w}'] = np.sqrt(
            cx.diff().rolling(ws, min_periods=1).var() +
            cy.diff().rolling(ws, min_periods=1).var()
        )

    # --- 6.1 更强的中心运动学时序 ---
    X = add_center_kinematics_v2(X, cx, cy, fps)

    # --- 7. 曲折度 / 多尺度 / 状态 / 长时程 / 累积路径 ---
    X = add_tortuosity_features(X, cx, cy, fps)
    X = add_curvature_features(X, cx, cy, fps)
    X = add_multiscale_features(X, cx, cy, fps)
    X = add_state_features(X, cx, cy, fps)
    X = add_longrange_features(X, cx, cy, fps)
    X = add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base=180)

    # --- 7.1 Arena spatial features --- ####
    if arena_width_cm is not None and arena_height_cm is not None:
        X = add_arena_spatial_features_single(
            X, cx, cy, fps,
            arena_w_cm=arena_width_cm,
            arena_h_cm=arena_height_cm,
            arena_shape=arena_shape or "square",
        )

    # --- 8. Groom + 解耦 ---
    X = add_groom_microfeatures(X, single_mouse, fps)
    X = add_body_part_dissociation(X, single_mouse, fps, available_body_parts)

    # --- 9. 过去vs未来非对称 & 分布漂移 ---
    X = add_speed_asymmetry_future_past_single(X, cx, cy, fps, horizon_base=30)
    X = add_gauss_shift_speed_future_past_single(X, cx, cy, fps, window_base=30)

    # --- 10. 关键部位微运动（nose/ears/tail_base） ---
    X = add_part_micro_motion_v2(X, single_mouse, available_body_parts, fps)

    # --- 11. 鼻-尾距离更密集时序 ---
    X = add_nt_dynamic_v2(X, single_mouse, available_body_parts, fps)

    # --- 12. 耳部动态 ---
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt(
            (single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
            (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2
        )

        # 扩展 offset 范围，增加更长时间跨度的对齐信息
        for off in [-40, -20, -10, 10, 20, 40]:
            o = _scale_signed(off, fps)
            X[f'ear_o{off}'] = ear_d.shift(-o)

        w = _scale(30, fps)
        X['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    # --- 13. 形态学/结构/频域代理 ---
    X = add_morphometric_pca(X, single_mouse, body_parts_tracked, fps)
    X = add_body_bend_angle(X, single_mouse, available_body_parts)
    X = add_frequency_jitter(X, single_mouse, available_body_parts, fps)

    # --- 14. 三角形面积 ---
    X = add_triangle_areas_single(X, single_mouse, available_body_parts)

    # --- 15. 关键特征的显式 lag/lead（轻量非因果上下文） ---
    if add_key_lags:
        if key_lag_features is None:
            # 最小集合：和你思考最一致的两类
            key_lag_features = [
                "nt_dist",
                "elong",
                "sp_m5",
                # 可选增强（二选一或只加一个）
                # "c_v",
                # "sp_ratio_5_40",
                # "c_v_contrast_15_90",
            ]
        X = add_key_feature_lags(
            X,
            fps=fps,
            keys=key_lag_features,
            lags_base=key_lags_base,
            add_future=True,
        )

    return X.astype(np.float32, copy=False)