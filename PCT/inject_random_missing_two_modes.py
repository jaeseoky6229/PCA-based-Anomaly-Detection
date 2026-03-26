import numpy as np

def inject_random_missing_two_modes(
    X,
    mode_axis=1,
    n_selected_modes=2,
    fraction=0.1,
    random_state=None
):
    """
    두 개(기본) 모드를 무작위 선택하고 그 모드 안에서만 MCAR 방식으로 결측(NaN) 부여.

    Parameters
    ----------
    X : np.ndarray
        원본 데이터 (수정하지 않음, 복사 후 처리).
    mode_axis : int, default=1
        '모드'(피처/변수)가 놓인 축 인덱스.
    n_selected_modes : int, default=2
        랜덤 선택할 모드 개수.
    fraction : float, default=0.1
        선택된 각 모드 내부에서 결측으로 바꿀 확률 (0~1).
    seed : int or None
        재현성 위한 시드.

    Returns
    -------
    X_missing : np.ndarray
        결측이 주입된 배열 (원본 복사본).
    selected_modes : np.ndarray
        결측이 부여된 모드 인덱스 (mode_axis 기준).
    missing_mask : np.ndarray (bool)
        True 위치가 새로 NaN으로 변환된 위치 (전체 shape 동일).
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    n_modes = X.shape[mode_axis]
    if n_selected_modes > n_modes:
        raise ValueError(f"n_selected_modes({n_selected_modes}) > n_modes({n_modes})")

    # 모드 인덱스 선택
    selected_modes = rng.choice(n_modes, size=n_selected_modes, replace=False)

    # 출력 준비
    X_missing = X.copy()
    missing_mask = np.zeros_like(X_missing, dtype=bool)

    # mode_axis 를 앞으로 이동해 처리 후 되돌림
    X_work = np.moveaxis(X_missing, mode_axis, 0)
    mask_work = np.moveaxis(missing_mask, mode_axis, 0)

    for m in selected_modes:
        # 해당 모드 슬라이스 shape: (기타 축들…)
        slice_shape = X_work[m].shape
        # MCAR 마스크
        m_mask = rng.random(slice_shape) < fraction
        # 기존 NaN 유지, 새로 True 인 곳만 기록
        new_mask = m_mask & ~np.isnan(X_work[m])
        X_work[m][new_mask] = np.nan
        mask_work[m][new_mask] = True

    # 축 원위치
    X_missing = np.moveaxis(X_work, 0, mode_axis)
    missing_mask = np.moveaxis(mask_work, 0, mode_axis)

    return X_missing