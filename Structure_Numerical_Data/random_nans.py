import numpy as np

# fraction 만큼 결측 비율 고정
def random_nans(data, fraction=0.2, random_state=42):
    """
    ndarray에 무작위로 결측값을 주입하는 함수
    
    Parameters:
    - data: numpy ndarray
    - fraction: 결측값 비율 (0~1)
    - random_state: 랜덤 시드
    
    Returns:
    - 결측값이 주입된 ndarray
    """
    np.random.seed(random_state)
    
    arr = data.copy()
    total_cells = arr.size
    num_nans = int(total_cells * fraction)
    
    # 전체 인덱스 리스트에서 무작위 샘플링
    indices = np.arange(total_cells)
    nan_indices = np.random.choice(indices, size=num_nans, replace=False)
    
    # 1차원 인덱스를 2차원(row, col)로 변환
    rows, cols = np.unravel_index(nan_indices, arr.shape)
    
    # ndarray에서는 직접 인덱싱으로 할당
    arr[rows, cols] = np.nan
    
    return arr


"""
결측 중복 허용 --> 결측비율변동
def random_nans(df, fraction=0.2, random_state = 42):
    np.random.seed(random_state)

    total_cells = df.size
    num_nans = int(total_cells * fraction)

    row_indices = np.random.randint(0, df.shape[0], num_nans)
    col_indices = np.random.randint(0, df.shape[1], num_nans)

    for r,c in zip(row_indices, col_indices):
        df.iat[r,c] = np.nan
    
    return df
"""