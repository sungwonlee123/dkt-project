import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def visualize_attention_patterns(model, dataset, student_id=None, sequence_length=10):
    """
    학습된 모델의 Attention 패턴을 시각화하는 함수
    Args:
        model: 학습된 SAKT 모델
        dataset: 데이터셋
        student_id: 특정 학생 ID (None이면 첫 번째 시퀀스 사용)
        sequence_length: 시각화할 시퀀스 길이 (기본값: 10)
    """
    # 임시로 시퀀스 길이 변경
    original_n = model.n
    original_P = model.P.clone()
    model.n = sequence_length
    model.P = torch.nn.Parameter(torch.Tensor(sequence_length, model.d))
    torch.nn.init.kaiming_normal_(model.P)
    
    # 패딩을 위한 준비
    pad_val = -1
    model.eval()
    
    # 시퀀스 준비 - 다양한 문제가 있는 시퀀스 찾기
    if student_id is None:
        print("\nSearching for a sequence with diverse problems...")
        best_seq_idx = 0
        max_unique_problems = 0
        
        # 처음 100개의 시퀀스에서 가장 다양한 문제가 있는 시퀀스 찾기
        for i in range(min(100, len(dataset.q_seqs))):
            valid_indices = dataset.q_seqs[i] != pad_val
            q_seq_temp = dataset.q_seqs[i][valid_indices]
            unique_problems = len(np.unique(q_seq_temp))
            
            if unique_problems > max_unique_problems and len(q_seq_temp) >= 5:
                max_unique_problems = unique_problems
                best_seq_idx = i
        
        print(f"\nFound sequence {best_seq_idx} with {max_unique_problems} unique problems")
        
        # 선택된 시퀀스의 데이터 가져오기
        valid_indices = dataset.q_seqs[best_seq_idx] != pad_val
        q_seq = dataset.q_seqs[best_seq_idx][valid_indices][:sequence_length]
        r_seq = dataset.r_seqs[best_seq_idx][valid_indices][:sequence_length]
        
        # feature가 있는 경우에만 가져오기
        has_diff = hasattr(dataset, 'd_seqs')
        has_attempts = hasattr(dataset, 'a_seqs')
        has_time = hasattr(dataset, 't_seqs')
        
        # 시퀀스 길이에 맞춰서 안전하게 인덱싱
        if has_diff:
            d_seq_full = dataset.d_seqs[best_seq_idx]
            try:
                d_seq = d_seq_full[valid_indices][:sequence_length]
            except IndexError:
                # 길이가 맞지 않으면 0으로 채움
                d_seq = np.zeros_like(q_seq)
        else:
            d_seq = np.zeros_like(q_seq)
            
        if has_attempts:
            a_seq_full = dataset.a_seqs[best_seq_idx]
            try:
                a_seq = a_seq_full[valid_indices][:sequence_length]
            except IndexError:
                a_seq = np.zeros_like(q_seq)
        else:
            a_seq = np.zeros_like(q_seq)
            
        if has_time:
            t_seq_full = dataset.t_seqs[best_seq_idx]
            try:
                t_seq = t_seq_full[valid_indices][:sequence_length]
            except IndexError:
                t_seq = np.zeros_like(q_seq)
        else:
            t_seq = np.zeros_like(q_seq)
        
        # 문제 ID를 실제 문제 이름으로 변환
        idx2q = {idx: q for q, idx in dataset.q2idx.items()}
        print("\nSelected problems:")
        for i, (q, r) in enumerate(zip(q_seq, r_seq)):
            print(f"{i+1}. Problem {idx2q[q]}: {'Correct' if r == 1 else 'Incorrect'}")
    else:
        # student_id로 해당 학생의 시퀀스 찾기
        student_idx = np.where(dataset.u_list == student_id)[0][0]
        valid_indices = dataset.q_seqs[student_idx] != pad_val
        q_seq = dataset.q_seqs[student_idx][valid_indices][:sequence_length]
        r_seq = dataset.r_seqs[student_idx][valid_indices][:sequence_length]
        d_seq = dataset.d_seqs[student_idx][valid_indices][:sequence_length]
        a_seq = dataset.a_seqs[student_idx][valid_indices][:sequence_length]
        t_seq = dataset.t_seqs[student_idx][valid_indices][:sequence_length]
    
    # 패딩 처리
    if len(q_seq) < sequence_length:
        padding_length = sequence_length - len(q_seq)
        q_seq = np.concatenate([q_seq, np.array([pad_val] * padding_length)])
        r_seq = np.concatenate([r_seq, np.array([pad_val] * padding_length)])
        d_seq = np.concatenate([d_seq, np.array([pad_val] * padding_length)])
        a_seq = np.concatenate([a_seq, np.array([pad_val] * padding_length)])
        t_seq = np.concatenate([t_seq, np.array([pad_val] * padding_length)])
    
    # 실제 문제 인덱스만 선택 (패딩 제외)
    valid_indices = q_seq != pad_val
    q_seq = q_seq[valid_indices]
    r_seq = r_seq[valid_indices]
    d_seq = d_seq[valid_indices]
    a_seq = a_seq[valid_indices]
    t_seq = t_seq[valid_indices]

    # 최대 10개의 문제만 선택
    max_display = min(10, len(q_seq))
    q_seq = q_seq[:max_display]
    r_seq = r_seq[:max_display]
    d_seq = d_seq[:max_display]
    a_seq = a_seq[:max_display]
    t_seq = t_seq[:max_display]

    # 텐서로 변환
    q = torch.LongTensor(q_seq).unsqueeze(0)
    r = torch.LongTensor(r_seq).unsqueeze(0)
    d = torch.FloatTensor(d_seq).unsqueeze(0)
    a = torch.FloatTensor(a_seq).unsqueeze(0)
    t = torch.FloatTensor(t_seq).unsqueeze(0)
    
    # Attention 가중치 얻기
    with torch.no_grad():
        try:
            # DKT2 모델의 경우
            if hasattr(model, 'feature_projection'):  # DKT2 모델의 특징
                s = d  # 정적 난이도는 동적 난이도와 동일
                _, attention_weights = model(q, r, q, d, s, a, t)
            # DKT3 또는 기본 SAKT 모델의 경우
            else:
                _, attention_weights = model(q, r, q, d, a, t)
        except TypeError as e:
            print(f"모델 호출 에러: {e}")
            # 기본 SAKT 모델의 경우 (최소 파라미터)
            _, attention_weights = model(q, r, q)
    
    # Attention 가중치를 numpy 배열로 변환
    attention_matrix = attention_weights[0].cpu().numpy()
    
    # 문제 이름 가져오기
    idx2q = {idx: q for q, idx in dataset.q2idx.items()}
    problem_names = [idx2q[idx] for idx in q_seq]
    
    # 어텐션 패턴 디버깅을 위한 출력
    print("Attention matrix shape before mean:", attention_matrix.shape)
    
    # head 별 평균 계산
    if len(attention_matrix.shape) == 3:  # [num_heads, seq_len, seq_len]
        attention_matrix = np.mean(attention_matrix, axis=0)
    print("Attention matrix shape after mean:", attention_matrix.shape)
    
    # 정규화
    if attention_matrix.max() != attention_matrix.min():
        attention_matrix = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min())
    
    # 문제 이름과 정답 여부 결합
    labels = [f"{name}\n({'O' if r_seq[i] == 1 else 'X'})" for i, name in enumerate(problem_names)]
    
    # 한글 폰트 설정
    import matplotlib.font_manager as fm
    import platform
    import matplotlib as mpl
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Linux':
        plt.rcParams['font.family'] = 'NanumGothic'
    
    mpl.rcParams['font.family'] = plt.rcParams['font.family']
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    plt.rcParams['font.size'] = 12  # 기본 폰트 크기 설정
    
    # Figure 1: 히트맵
    fig1, ax1 = plt.subplots(figsize=(15, 12))  # 세로 크기 증가
    
    # 문제 이름 한글 변환
    problem_translations = {
        'Equivalent Fractions': '동치분수',
        'Circle Graph': '원그래프',
        'Proportion': '비례',
        'Finding Percents': '백분율',
        'Probability of Two Distinct Events': '두 사건의 확률',
        'Pythagorean Theorem': '피타고라스 정리'
    }
    
    # 레이블에 한글 추가
    new_labels = [f"{name}\n{problem_translations.get(name.split('(')[0].strip(), '')}\n({'O' if r_seq[i] == 1 else 'X'})" 
                 for i, name in enumerate(problem_names)]
    
    sns.heatmap(attention_matrix, 
                xticklabels=new_labels,
                yticklabels=new_labels,
                cmap='YlOrRd',
                annot=True,  # 값 표시
                fmt='.2f',   # 소수점 2자리
                square=True,  # 정사각형 셀
                cbar_kws={'label': 'Attention Score (영향력 점수)'},
                ax=ax1)
    
    ax1.set_title('문제 간 영향력 분석\nAttention Pattern Analysis\nHow much each problem influences others', pad=20)
    ax1.set_xlabel('이전 문제 (Previous Problems)', labelpad=10)
    ax1.set_ylabel('현재 문제 (Current Problem)', labelpad=10)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # 범례 추가
    legend_text = """
    해석 방법:
    1. 세로축: 현재 풀고 있는 문제
    2. 가로축: 이전에 푼 문제들
    3. 값 범위: 0 (영향 없음) ~ 1 (최대 영향)
    4. O: 맞은 문제, X: 틀린 문제
    5. 색이 진할수록 영향력이 큼
    """
    plt.figtext(1.02, 0.5, legend_text, fontsize=10, va='center')
    
    plt.tight_layout()
    
    # Figure 2: 그래프 시각화
    fig2, ax2 = plt.subplots(figsize=(15, 10))
    G = nx.DiGraph()
    
    # 문제 이름 한글 변환
    problem_translations = {
        'Equivalent Fractions': '동치분수',
        'Circle Graph': '원그래프',
        'Proportion': '비례',
        'Finding Percents': '백분율',
        'Probability of Two Distinct Events': '두 사건의 확률',
        'Pythagorean Theorem': '피타고라스 정리'
    }
    
    # 노드와 노드 속성을 저장할 딕셔너리
    node_labels = {}
    node_colors = {}
    
    # 디버깅을 위한 출력
    print("\nNode creation details:")
    
    # 노드 추가 (이름 줄이기)
    for i, name in enumerate(problem_names):
        # 원본 이름에서 (O) 또는 (X) 제거
        base_name = name.split('(')[0].strip()
        # 한글 이름 가져오기
        kor_name = problem_translations.get(base_name, base_name)
        # 정답 여부 확인
        is_correct = r_seq[i] == 1
        # 문제 번호와 정답 여부를 포함한 고유 ID 생성
        node_id = f"{i+1}_{base_name}"
        # 노드 레이블 생성 (문제 번호, 한글 이름, 정답 여부 포함)
        label = f"{i+1}. {kor_name}\n{'(O)' if is_correct else '(X)'}"
        # 정답 여부에 따른 노드 색상
        color = 'lightgreen' if is_correct else 'lightcoral'
        # 노드 추가 및 속성 저장
        G.add_node(node_id)
        node_labels[node_id] = label
        node_colors[node_id] = color
        
        # 디버깅 정보 출력
        print(f"Node {i+1}:")
        print(f"  ID: {node_id}")
        print(f"  Label: {label}")
        print(f"  Color: {color}")
        
        # 디버깅 정보 출력
        print(f"Node {i+1}:")
        print(f"  Original name: {name}")
        print(f"  Korean name: {kor_name}")
        print(f"  Correct: {is_correct}")
        print(f"  Label: {label}")
        print(f"  Color: {color}")
    
    # 엣지 추가 (상위 70% 엣지 표시)
    flat_attention = attention_matrix.flatten()
    threshold = np.percentile(flat_attention[flat_attention > 0], 30)  # 상위 70% 엣지
    
    edge_weights = []
    edge_colors = []
    
    for i in range(len(problem_names)):
        for j in range(len(problem_names)):
            if attention_matrix[i][j] > threshold and i != j:
                source_id = f"{i+1}_{problem_names[i]}"
                target_id = f"{j+1}_{problem_names[j]}"
                G.add_edge(source_id, target_id, 
                          weight=attention_matrix[i][j])
                edge_weights.append((attention_matrix[i][j] - threshold) * 10)  # 상대적 두께
                edge_colors.append('gray')
    
    # 그래프 레이아웃 설정 (원형 레이아웃으로 변경)
    pos = nx.circular_layout(G)
    
    # 그래프 그리기
    nx.draw(G, pos, 
            with_labels=True,
            labels=node_labels,  # 한글 레이블 사용
            node_color=[node_colors[node] for node in G.nodes()],  # 노드 순서에 맞게 색상 리스트 생성
            node_size=4000,  # 노드 크기 증가
            font_size=11,    # 폰트 크기 증가
            font_weight='bold',
            font_family='AppleGothic',  # 명시적으로 폰트 지정
            width=edge_weights,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=20,
            ax=ax2,
            connectionstyle="arc3,rad=0.2")
    
    # 한글 범례 추가 (폰트 명시)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=15, label='정답 (O)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                  markersize=15, label='오답 (X)')
    ]
    
    # 범례 위치 조정 및 추가
    legend = ax2.legend(handles=legend_elements, 
                       title='문제 풀이 결과',
                       loc='center left', 
                       bbox_to_anchor=(1.2, 0.5),
                       fontsize=11,
                       title_fontsize=12)
    # 범례 폰트 설정
    plt.setp(legend.get_texts(), family='AppleGothic')
    
    # 추가 설명 텍스트
    description = """
    그래프 해석 방법:
    1. 노드(원): 각각의 문제를 나타냄
    2. 노드 색상: 초록색은 정답, 빨간색은 오답
    3. 화살표: 문제들 간의 영향 관계를 표시
    4. 화살표 두께: 영향력의 강도 (두꺼울수록 강한 영향)
    5. 화살표 방향: 이전 문제가 다음 문제에 미치는 영향

    Graph Interpretation:
    1. Node: Represents each problem
    2. Node Color: Green for correct, Red for incorrect
    3. Arrow: Shows influence between problems
    4. Arrow Thickness: Strength of influence
    5. Arrow Direction: Impact from previous to next problem
    """
    # 설명 텍스트 위치 조정
    plt.figtext(1.02, 0.5, description, fontsize=10, va='center')
    
    # 제목 수정 (한글/영문)
    ax2.set_title('문제 간 관계 그래프\nProblem Relationship Graph', pad=20)
    
    plt.tight_layout()
    
    # 원래 모델 설정 복구
    model.n = original_n
    model.P = torch.nn.Parameter(original_P)
    
    return fig1, fig2  # 두 figure 모두 반환

def plot_learning_progress(model, dataset, student_id=None, window_size=5, sequence_length=10):
    """
    학습 진행 상황을 시각화하는 함수
    Args:
        model: 학습된 SAKT 모델
        dataset: 데이터셋
        student_id: 특정 학생 ID (None이면 첫 번째 학생)
        window_size: 이동 평균 윈도우 크기
        sequence_length: 시각화할 시퀀스 길이
    """
    # 시퀀스 길이 임시 변경
    original_n = model.n
    original_P = model.P.clone()
    model.n = sequence_length
    model.P = torch.nn.Parameter(torch.Tensor(sequence_length, model.d))
    torch.nn.init.kaiming_normal_(model.P)
    
    # 학생 데이터 가져오기
    if student_id is None:
        q_seq = dataset.q_seqs[0]
        r_seq = dataset.r_seqs[0]
    else:
        student_idx = np.where(dataset.u_list == student_id)[0][0]
        q_seq = dataset.q_seqs[student_idx]
        r_seq = dataset.r_seqs[student_idx]
    
    # 실제 정답률 계산 (이동 평균)
    correct_rates = np.convolve(r_seq, np.ones(window_size)/window_size, mode='valid')
    
    # 모델 예측
    model.eval()
    with torch.no_grad():
        q = torch.LongTensor(q_seq).unsqueeze(0)
        r = torch.LongTensor(r_seq).unsqueeze(0)
        
        try:
            # DKT2 또는 DKT3 모델의 경우
            if hasattr(model, 'feature_projection') or hasattr(dataset, 'd_seqs'):
                has_diff = hasattr(dataset, 'd_seqs')
                has_static_diff = hasattr(dataset, 's_seqs')
                has_attempts = hasattr(dataset, 'a_seqs')
                has_time = hasattr(dataset, 't_seqs')
                
                student_idx = 0 if student_id is None else np.where(dataset.u_list == student_id)[0][0]
                d_seq = dataset.d_seqs[student_idx] if has_diff else np.zeros_like(q_seq)
                s_seq = dataset.s_seqs[student_idx] if has_static_diff else d_seq
                a_seq = dataset.a_seqs[student_idx] if has_attempts else np.zeros_like(q_seq)
                t_seq = dataset.t_seqs[student_idx] if has_time else np.zeros_like(q_seq)
                
                d = torch.FloatTensor(d_seq).unsqueeze(0)
                s = torch.FloatTensor(s_seq).unsqueeze(0)
                a = torch.FloatTensor(a_seq).unsqueeze(0)
                t = torch.FloatTensor(t_seq).unsqueeze(0)
                
                if hasattr(model, 'feature_projection'):  # DKT2 모델의 경우
                    pred_probs, _ = model(q, r, q, d, s, a, t)
                else:  # DKT3 모델의 경우
                    pred_probs, _ = model(q, r, q, d, a, t)
            # 기본 SAKT 모델의 경우 (최소 파라미터)
            else:
                pred_probs, _ = model(q, r, q)
        except TypeError as e:
            print(f"모델 호출 에러: {e}, 기본 파라미터로 시도합니다.")
            # 항상 동작하는 기본 호출
            pred_probs, _ = model(q, r, q)
        
        pred_probs = pred_probs.cpu().numpy()
    
    # 예측값의 이동 평균
    pred_rates = np.convolve(pred_probs, np.ones(window_size)/window_size, mode='valid')
    
    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    x = np.arange(len(correct_rates))
    
    plt.plot(x, correct_rates, 'b-', label='Actual Performance', alpha=0.7)
    plt.plot(x, pred_rates, 'r--', label='Predicted Performance', alpha=0.7)
    
    plt.title('Learning Progress Over Time')
    plt.xlabel('Problem Sequence')
    plt.ylabel('Performance (Moving Average)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 원래 설정 복구
    model.n = original_n
    model.P = original_P
    
    return plt.gcf()

def analyze_skill_mastery(model, dataset, student_id=None):
    """
    스킬 숙련도를 분석하고 시각화하는 함수
    Args:
        model: 학습된 SAKT 모델
        dataset: 데이터셋
        student_id: 특정 학생 ID (None이면 첫 번째 학생)
    """
    plt.figure(figsize=(12, 6))
    # 학생 데이터 가져오기
    if student_id is None:
        q_seq = dataset.q_seqs[0]
        r_seq = dataset.r_seqs[0]
    else:
        student_idx = np.where(dataset.u_list == student_id)[0][0]
        q_seq = dataset.q_seqs[student_idx]
        r_seq = dataset.r_seqs[student_idx]
    
    # 각 문제 유형별 성과 계산
    skill_performance = {}
    idx2q = {idx: q for q, idx in dataset.q2idx.items()}
    
    for q, r in zip(q_seq, r_seq):
        skill_name = idx2q[q]
        if skill_name not in skill_performance:
            skill_performance[skill_name] = {'correct': 0, 'total': 0}
        skill_performance[skill_name]['total'] += 1
        if r == 1:
            skill_performance[skill_name]['correct'] += 1
    
    # 숙련도 계산
    skill_mastery = {skill: stats['correct']/stats['total'] 
                     for skill, stats in skill_performance.items()}
    
    # 시각화
    plt.figure(figsize=(15, 8))
    skills = list(skill_mastery.keys())
    mastery_values = list(skill_mastery.values())
    
    # 막대 그래프
    bars = plt.bar(range(len(skills)), mastery_values)
    
    # 색상 설정
    colors = ['#ff9999' if v < 0.6 else '#66b3ff' if v < 0.8 else '#99ff99' for v in mastery_values]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.title('Skill Mastery Analysis')
    plt.xlabel('Skills')
    plt.ylabel('Mastery Level (0-1)')
    plt.xticks(range(len(skills)), skills, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # 숙련도 레벨 표시
    for i, v in enumerate(mastery_values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()
