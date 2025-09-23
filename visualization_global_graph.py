import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np
from tqdm import tqdm
import inspect

def visualize_global_relationship_graph(model, dataset, top_n=15):
    """
    전체 데이터셋에 대한 문제 간 관계를 그래프로 시각화하는 함수
    Args:
        model: 학습된 SAKT 모델
        dataset: 데이터셋
        top_n: 표시할 상위 문제 개수 (기본값: 15)
    """
    print("\nAnalyzing global problem relationships...")
    
    # 모델 설정
    model.eval()
    pad_val = -1
    
    # 전체 문제 유형에 대한 attention matrix 초기화
    problem_attention_sum = np.zeros((dataset.num_q, dataset.num_q))
    problem_attention_count = np.zeros((dataset.num_q, dataset.num_q))
    problem_correct_count = {i: {'correct': 0, 'total': 0} for i in range(dataset.num_q)}
    
    # 전체 시퀀스에 대해 attention 패턴 분석
    seq_lens = [
        len(dataset.q_seqs),
        len(dataset.r_seqs)
    ]
    if hasattr(dataset, 'd_seqs'):
        seq_lens.append(len(dataset.d_seqs))
    if hasattr(dataset, 's_seqs'):
        seq_lens.append(len(dataset.s_seqs))
    if hasattr(dataset, 'a_seqs'):
        seq_lens.append(len(dataset.a_seqs))
    if hasattr(dataset, 't_seqs'):
        seq_lens.append(len(dataset.t_seqs))
    min_seq_len = min(seq_lens)
    for i in tqdm(range(min_seq_len), desc="Processing sequences"):
        # 시퀀스 준비 - 원래 길이 유지
        q_seq = dataset.q_seqs[i]
        r_seq = dataset.r_seqs[i]
        has_diff = hasattr(dataset, 'd_seqs')
        has_static_diff = hasattr(dataset, 's_seqs')
        has_attempts = hasattr(dataset, 'a_seqs')
        has_time = hasattr(dataset, 't_seqs')
        d_seq = dataset.d_seqs[i] if has_diff else np.zeros_like(q_seq)
        s_seq = dataset.s_seqs[i] if has_static_diff else d_seq
        a_seq = dataset.a_seqs[i] if has_attempts else np.zeros_like(q_seq)
        t_seq = dataset.t_seqs[i] if has_time else np.zeros_like(q_seq)
        # 패딩된 부분 제거 및 길이 맞추기
        valid_indices = q_seq != pad_val
        min_len = min(len(q_seq), len(r_seq), len(d_seq), len(s_seq), len(a_seq), len(t_seq), len(valid_indices))
        q_seq = q_seq[:min_len][valid_indices[:min_len]]
        r_seq = r_seq[:min_len][valid_indices[:min_len]]
        d_seq = d_seq[:min_len][valid_indices[:min_len]]
        s_seq = s_seq[:min_len][valid_indices[:min_len]]
        a_seq = a_seq[:min_len][valid_indices[:min_len]]
        t_seq = t_seq[:min_len][valid_indices[:min_len]]
        
        # 정답률 계산을 위한 카운트
        for q, r in zip(q_seq, r_seq):
            if q != pad_val:
                problem_correct_count[q]['total'] += 1
                if r == 1:
                    problem_correct_count[q]['correct'] += 1
        
        if len(q_seq) < 2:  # 최소 2개의 문제가 필요
            continue
        
        # Attention 패턴 계산
        with torch.no_grad():
            window_size = 10  # 시각화를 위한 적절한 크기
            if len(q_seq) < window_size:
                continue
                
            window_q = q_seq[:window_size]
            window_r = r_seq[:window_size]
            window_d = d_seq[:window_size]
            window_s = s_seq[:window_size]
            window_a = a_seq[:window_size]
            window_t = t_seq[:window_size]
            
            # 패딩 처리
            padding_length = model.n - window_size
            q_pad = np.pad(window_q, (0, padding_length), constant_values=0)  # 0을 사용하여 패딩
            r_pad = np.pad(window_r, (0, padding_length), constant_values=0)
            d_pad = np.pad(window_d, (0, padding_length), constant_values=0.0)
            s_pad = np.pad(window_s, (0, padding_length), constant_values=0.0)
            a_pad = np.pad(window_a, (0, padding_length), constant_values=0.0)
            t_pad = np.pad(window_t, (0, padding_length), constant_values=0.0)
            
            # 모델에 필요한 텐서만 준비
            q = torch.LongTensor(q_pad).unsqueeze(0)
            r = torch.LongTensor(r_pad).unsqueeze(0)
            d = torch.FloatTensor(d_pad).unsqueeze(0) if has_diff else None
            s = torch.FloatTensor(s_pad).unsqueeze(0) if has_static_diff else None
            a = torch.FloatTensor(a_pad).unsqueeze(0) if has_attempts else None
            t = torch.FloatTensor(t_pad).unsqueeze(0) if has_time else None
            
            # 모델의 forward 메소드 시그니처를 동적으로 확인하여 적절한 파라미터로 호출
            try:
                # forward 메소드의 파라미터 확인
                forward_signature = inspect.signature(model.forward)
                param_names = list(forward_signature.parameters.keys())
                
                # 파라미터 개수와 이름에 따라 적절히 호출
                if len(param_names) >= 7:  # DKT2: (q, r, qry, d_diff, s_diff, att, time)
                    # None 체크 및 기본값 할당
                    d = d if d is not None else torch.zeros_like(q, dtype=torch.float32)
                    s = s if s is not None else torch.zeros_like(q, dtype=torch.float32)
                    a = a if a is not None else torch.zeros_like(q, dtype=torch.float32)
                    t = t if t is not None else torch.zeros_like(q, dtype=torch.float32)
                    _, curr_attention = model(q, r, q, d, s, a, t)
                elif 'd_diff' in param_names or len(param_names) == 4:
                    # SLAM with difficulty (d_diff 파라미터 있음)
                    d = d if d is not None else torch.zeros_like(q, dtype=torch.float32)
                    _, curr_attention = model(q, r, q, d)
                else:
                    # SAKT/DKT-old (기본 모델: q, r, qry)
                    _, curr_attention = model(q, r, q)
            except (TypeError, RuntimeError) as e:
                print(f"모델 호출 에러: {e}, 기본 파라미터로 재시도합니다.")
                try:
                    _, curr_attention = model(q, r, q)
                except Exception as e2:
                    print(f"기본 모델 호출 실패: {e2}")
                    continue
            
            curr_attention = curr_attention[0].cpu().detach().numpy()
            
            # head 별 평균 계산
            if len(curr_attention.shape) == 3:
                curr_attention = np.mean(curr_attention, axis=0)
            
            # 유효한 부분만 사용
            valid_attention = curr_attention[:len(q_seq), :len(q_seq)]
            
            # 문제 유형별 attention 누적
            for idx1 in range(window_size):
                for idx2 in range(window_size):
                    prob1 = int(window_q[idx1])  # 현재 문제
                    prob2 = int(window_q[idx2])  # 이전 문제
                    if 0 <= prob1 < dataset.num_q and 0 <= prob2 < dataset.num_q:
                        problem_attention_sum[prob1, prob2] += valid_attention[idx1, idx2]
                        problem_attention_count[prob1, prob2] += 1
    
    # 문제 유형별 등장 횟수 계산
    problem_freq = {i: problem_correct_count[i]['total'] for i in range(dataset.num_q)}
    
    # 가장 많이 등장하는 상위 N개 문제 유형 선택
    top_problems = sorted(problem_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_problem_indices = [idx for idx, _ in top_problems]
    
    # 선택된 문제들에 대한 attention matrix 추출
    attention_matrix = np.divide(
        problem_attention_sum[top_problem_indices][:, top_problem_indices],
        problem_attention_count[top_problem_indices][:, top_problem_indices],
        out=np.zeros((len(top_problem_indices), len(top_problem_indices))),
        where=problem_attention_count[top_problem_indices][:, top_problem_indices]!=0
    )
    
    # 정규화
    if attention_matrix.max() != attention_matrix.min():
        attention_matrix = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min())
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Figure: 그래프
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # 문제 이름 한글 변환
    problem_translations = {
        'Equation Solving Two or Fewer Steps': '2단계 이하 방정식 풀이',
        'Conversion of Fraction Decimals Percents': '분수/소수/백분율 변환',
        'Addition and Subtraction Integers': '정수의 덧셈과 뺄셈',
        'Addition and Subtraction Fractions': '분수의 덧셈과 뺄셈',
        'Percent Of': '백분율',
        'Pattern': '패턴',
        'Dividing Fractions': '분수의 나눗셈',
        'Equation Solving More Than Two Steps': '2단계 이상 방정식 풀이',
        'Probability of a Single Event': '한 사건의 확률',
        'Probability of Two Distinct Events': '두 사건의 확률',
        'Subtraction Whole Numbers': '자연수의 뺄셈',
        'Absolute Value': '절댓값',
        'Ordering Positive Decimals': '양의 소수 크기 비교'
    }
    
    # 그래프 생성
    G = nx.DiGraph()
    
    # 노드와 노드 속성 저장
    node_labels = {}
    node_colors = {}
    
    # 노드 추가
    idx2q = {idx: q for q, idx in dataset.q2idx.items()}
    for idx in top_problem_indices:
        name = idx2q[idx]
        kor_name = problem_translations.get(name.split('(')[0].strip(), name)
        stats = problem_correct_count[idx]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        freq = problem_freq[idx]
        
        # 노드 ID와 레이블 생성
        node_id = str(idx)
        label = f"{kor_name}\n(출현: {freq}회\n정답률: {accuracy:.0%})"
        
        # 노드 추가 및 속성 저장
        G.add_node(node_id)
        node_labels[node_id] = label
        node_colors[node_id] = plt.cm.RdYlGn(accuracy)  # 정답률에 따른 색상
    
    # attention 값이 0.3 이상인 연결만 표시
    flat_attention = attention_matrix.flatten()
    threshold = 0.3  # 절대값 기준으로 설정
    
    edge_weights = []
    edge_colors = []
    
    # 엣지 추가
    for i in range(len(top_problem_indices)):
        for j in range(len(top_problem_indices)):
            if attention_matrix[i][j] > threshold and i != j:
                source_id = str(top_problem_indices[i])
                target_id = str(top_problem_indices[j])
                weight = attention_matrix[i][j]
                
                G.add_edge(source_id, target_id, weight=weight)
                # 약한 연결도 보이도록 가중치 조정
                scaled_weight = (weight - threshold) / (attention_matrix.max() - threshold)
                
                # 모든 연결의 두께를 강도에 비례하도록 설정
                edge_weights.append(max(0.5, scaled_weight * 3))  # 최소 두께 0.5 보장
                
                # 연결 강도에 따른 색상 지정
                if weight < 0.5:
                    edge_colors.append(plt.cm.Greys(0.5))  # 0.3~0.5는 회색
                else:
                    edge_colors.append(plt.cm.Reds(scaled_weight))  # 0.5 이상은 빨간색 계열
    
    # 그래프 레이아웃 설정
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # 그래프 그리기 - 약한 연결도 보이도록 설정 수정
    nx.draw(G, pos,
            with_labels=True,
            labels=node_labels,
            node_color=[node_colors[node] for node in G.nodes()],
            node_size=5500,  # 노드 크기 조정
            font_size=9,
            font_weight='bold',
            font_family='AppleGothic',
            width=edge_weights,
            edge_color=[color for color in edge_colors],
            arrows=True,
            arrowsize=15,    # 화살표 크기 줄임
            ax=ax,
            connectionstyle="arc3,rad=0.1",  # 곡선 정도 줄임
            alpha=0.8)       # 투명도 조정
    
    # 제목 설정
    plt.title('전체 학생 데이터 기반 문제 간 관계 그래프', pad=20, fontsize=14)
    
    # 범례 추가
    legend_text = """
    < 그래프 해석 방법 >
    
    1. 노드(원) 의미
       ∙ 크기: 모든 노드 동일
       ∙ 색상: 정답률 (녹색 높음, 빨강 낮음)
       ∙ 레이블: 문제 유형, 출현 횟수, 정답률
    
    2. 화살표 의미
       ∙ 방향: 이전 문제 → 현재 문제
       ∙ 굵기: 영향력 강도
       ∙ 색상: 영향력 강도 (진할수록 강함)
    
    3. 해석 포인트
       ∙ 굵고 진한 화살표: 강한 학습 연관성
       ∙ 노드 군집: 서로 관련된 문제들
       ∙ 화살표 방향: 학습 순서 추천
    
    * 이 그래프는 전체 학생들(총 {num_students}명)의
      상위 {top_n}개 빈출 문제 유형만 표시합니다.
    """.format(num_students=dataset.num_u, top_n=top_n)
    
    # 범례 위치 조정 및 추가
    plt.figtext(1.02, 0.5, legend_text,
                fontsize=11,
                va='center',
                bbox=dict(facecolor='white',
                         edgecolor='lightgray',
                         boxstyle='round,pad=1'))
    
    # 그래프 크기와 여백 조정
    plt.gcf().set_size_inches(15, 12)
    plt.tight_layout()
    
    return fig
