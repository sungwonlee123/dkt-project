import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_slam_attention_patterns_enhanced(model, dataset, sequence_length=None, basic_matrix=None, original_matrix=None, problem_names=None):
    """
    SLAM 데이터셋 전용 개선된 Attention 패턴 시각화 함수 (Old SLAM, Self-Attention SLAM)
    - Basic 결과를 가져와서 정규화 전/후 비교 표시
    """
    print("\nAnalyzing enhanced SLAM attention patterns...")
    
    if basic_matrix is None or original_matrix is None:
        print("ERROR: Both basic_matrix and original_matrix required for Enhanced visualization")
        return None, None, None
    
    print("Using Basic attention matrices for Enhanced visualization (ensuring consistency)")
    print(f"Received problem names: {problem_names[:3] if problem_names else 'None'}...")  # 처음 3개만 출력
    
    # Basic에서 가져온 매트릭스들
    normalized_matrix = basic_matrix.copy()  # 정규화된 매트릭스 (0~1 범위)
    original_matrix_real = original_matrix.copy()  # 실제 원본 매트릭스
    
    
    print(f"\nSLAM 개선된 어텐션 값 분석:")
    print(f"원본 최솟값: {original_matrix_real.min():.6f}")
    print(f"원본 최댓값: {original_matrix_real.max():.6f}")
    print(f"정규화 최솟값: {normalized_matrix.min():.6f}")
    print(f"정규화 최댓값: {normalized_matrix.max():.6f}")
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Basic에서 전달받은 매트릭스는 이미 상위 15개로 축소된 상태
    matrix_size = normalized_matrix.shape[0]  # Basic에서 전달받은 매트릭스 크기
    print(f"Basic에서 전달받은 매트릭스 크기: {matrix_size}x{matrix_size}")
    
    # 단순히 0~matrix_size 인덱스 사용
    top_problem_indices = list(range(matrix_size))
    
    # Basic에서 전달받은 매트릭스 그대로 사용
    top_attention_original = original_matrix_real  # 진짜 원본값
    top_attention_normalized = normalized_matrix   # 정규화된 값
    
    # 문제 이름 사용 (Basic에서 전달받은 것 사용)
    if problem_names is None:
        problem_names = []
        for i in range(matrix_size):
            problem_names.append(f'문제{i+1}\\n(상위{i+1}번째)')
    
    # 이중 히트맵 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # 원본 값 히트맵 (왼쪽) - SLAM 전용 범위
    im1 = sns.heatmap(top_attention_original, 
                      annot=True, 
                      fmt='.3f',
                      cmap='YlOrRd',  # 빨강-주황 계열 사용
                      xticklabels=problem_names,
                      yticklabels=problem_names,
                      ax=ax1,
                      cbar_kws={'label': '원본 어텐션 값'},
                      square=True,
                      vmin=0,  # 최솟값 고정
                      vmax=top_attention_original.max())  # 원본 최댓값
    
    ax1.set_title('어텐션 영향력 (원본 값) - SLAM', pad=20, fontsize=14)
    ax1.set_xlabel('이전에 푼 문제', labelpad=10)
    ax1.set_ylabel('현재 풀고 있는 문제', labelpad=10)
    
    # X축 라벨 대각선으로 회전 (Basic처럼)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # 정규화된 값 히트맵 (오른쪽) - 0~1 범위 고정
    im2 = sns.heatmap(top_attention_normalized, 
                      annot=True, 
                      fmt='.3f',
                      cmap='YlOrRd',  # 빨강-주황 계열 사용
                      xticklabels=problem_names,
                      yticklabels=problem_names,
                      ax=ax2,
                      cbar_kws={'label': '정규화된 어텐션 값 (0~1)'},
                      square=True,
                      vmin=0,  # 0으로 고정
                      vmax=1)  # 1로 고정
    
    ax2.set_title('어텐션 영향력 (정규화) - SLAM', pad=20, fontsize=14)
    ax2.set_xlabel('이전에 푼 문제', labelpad=10)
    ax2.set_ylabel('현재 풀고 있는 문제', labelpad=10)
    
    # X축 라벨 대각선으로 회전 (Basic처럼)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    # 전체 제목
    fig.suptitle('SLAM 데이터셋: 전체 학생 데이터 기반 문제 간 영향력 분석 (원본 vs 정규화)', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    return fig, original_matrix_real, normalized_matrix
