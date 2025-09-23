import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import inspect

def visualize_slam_attention_patterns(model, dataset, sequence_length=100, return_matrix=False):
    """
    SLAM 데이터셋 전용 Attention 패턴 시각화 함수 (Old SLAM, Self-Attention SLAM)
    Args:
        model: 학습된 SAKT 모델
        dataset: 데이터셋
        sequence_length: 시각화할 시퀀스 길이 (기본값: 100)
        return_matrix: 매트릭스 반환 여부
    """
    print("\nAnalyzing SLAM attention patterns...")
    
    # 모델 설정
    model.eval()
    pad_val = -1
    
    # 문제 유형별 정답/오답 카운트 및 누적 attention 저장
    problem_stats = {}
    attention_sum = None
    attention_count = 0
    
    # 전체 문제 유형에 대한 attention matrix 초기화
    problem_attention_sum = np.zeros((dataset.num_q, dataset.num_q))
    problem_attention_count = np.zeros((dataset.num_q, dataset.num_q))
    problem_correct_count = {i: {'correct': 0, 'total': 0} for i in range(dataset.num_q)}
    
    # 전체 시퀀스에 대해 attention 패턴 분석
    seq_lens = [len(dataset.q_seqs), len(dataset.r_seqs)]
    if hasattr(dataset, 'd_seqs'):
        seq_lens.append(len(dataset.d_seqs))
    if hasattr(dataset, 's_seqs'):
        seq_lens.append(len(dataset.s_seqs))
    if hasattr(dataset, 'a_seqs'):
        seq_lens.append(len(dataset.a_seqs))
    if hasattr(dataset, 't_seqs'):
        seq_lens.append(len(dataset.t_seqs))
    
    min_seq_len = min(seq_lens)
    for i in tqdm(range(min_seq_len), desc="Processing SLAM sequences"):
        # 시퀀스 준비
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
        
        # 패딩된 부분 제거
        valid_indices = q_seq != pad_val
        seq_len = valid_indices.sum()
        min_len = min(len(q_seq), len(r_seq), len(d_seq), len(s_seq), len(a_seq), len(t_seq), len(valid_indices))
        q_seq = q_seq[:min_len][valid_indices[:min_len]]
        r_seq = r_seq[:min_len][valid_indices[:min_len]]
        d_seq = d_seq[:min_len][valid_indices[:min_len]]
        s_seq = s_seq[:min_len][valid_indices[:min_len]]
        a_seq = a_seq[:min_len][valid_indices[:min_len]]
        t_seq = t_seq[:min_len][valid_indices[:min_len]]
        
        # 시퀀스 길이 조정
        window_size = min(10, seq_len)
        if seq_len < 2:
            continue
        if seq_len < window_size:
            pad_len = window_size - seq_len
            q_seq = np.pad(q_seq, (0, pad_len), constant_values=0)
            r_seq = np.pad(r_seq, (0, pad_len), constant_values=0)
            d_seq = np.pad(d_seq, (0, pad_len), constant_values=0.0)
            s_seq = np.pad(s_seq, (0, pad_len), constant_values=0.0)
            a_seq = np.pad(a_seq, (0, pad_len), constant_values=0.0)
            t_seq = np.pad(t_seq, (0, pad_len), constant_values=0.0)
        else:
            q_seq = q_seq[:window_size]
            r_seq = r_seq[:window_size]
            d_seq = d_seq[:window_size]
            s_seq = s_seq[:window_size]
            a_seq = a_seq[:window_size]
            t_seq = t_seq[:window_size]
        
        # 정답률 계산을 위한 카운트
        for q, r in zip(q_seq, r_seq):
            if q != pad_val:
                problem_correct_count[q]['total'] += 1
                if r == 1:
                    problem_correct_count[q]['correct'] += 1
        
        if len(q_seq) < 2:
            continue
            
        # 문제 유형별 정답/오답 카운트
        idx2q = {idx: q for q, idx in dataset.q2idx.items()}
        for q, r in zip(q_seq, r_seq):
            problem_name = idx2q[q]
            if problem_name not in problem_stats:
                problem_stats[problem_name] = {'correct': 0, 'total': 0}
            problem_stats[problem_name]['total'] += 1
            if r == 1:
                problem_stats[problem_name]['correct'] += 1
        
        # 시퀀스를 작은 윈도우로 처리
        window_size = 10
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
        q_pad = np.pad(window_q, (0, padding_length), constant_values=0)
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
        
        # 입력 텐서 준비 및 타입 변환
        q = q.long()
        r = r.long()
        
        # SLAM 전용 모델 호출 (Old SLAM vs Self-Attention SLAM)
        try:
            # forward 메소드의 파라미터 확인
            forward_signature = inspect.signature(model.forward)
            param_names = list(forward_signature.parameters.keys())
            
            # SLAM 모델 구분: difficulty 파라미터 있는지 확인
            if 'd_diff' in param_names and len(param_names) == 4:
                # SLAM with difficulty (4개 파라미터)
                _, curr_attention = model(q, r, q, d)
            else:
                # Old SLAM or Self-Attention SLAM (3개 파라미터)
                _, curr_attention = model(q, r, q)
                
        except (TypeError, RuntimeError) as e:
            # 에러 발생시 기본 3-parameter로 fallback
            try:
                _, curr_attention = model(q, r, q)
            except Exception as e2:
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
                prob1 = int(window_q[idx1])
                prob2 = int(window_q[idx2])
                
                if prob1 < dataset.num_q and prob2 < dataset.num_q:
                    problem_attention_sum[prob1, prob2] += valid_attention[idx1, idx2]
                    problem_attention_count[prob1, prob2] += 1
        
        # 전체 attention 누적
        if attention_sum is None:
            attention_sum = valid_attention
            attention_count = 1
        else:
            min_size = min(attention_sum.shape[0], valid_attention.shape[0])
            attention_sum = attention_sum[:min_size, :min_size] + valid_attention[:min_size, :min_size]
            attention_count += 1
    
    # 평균 계산
    problem_attention_count[problem_attention_count == 0] = 1
    attention_matrix = problem_attention_sum / problem_attention_count
    
    # 문제 빈도 계산
    problem_freq = {}
    for i in range(dataset.num_q):
        problem_freq[i] = np.sum(problem_attention_count[i, :])
    
    print(f"\nSLAM 데이터셋 어텐션 값 분석:")
    print(f"최솟값: {attention_matrix.min():.6f}")
    print(f"최댓값: {attention_matrix.max():.6f}")
    print(f"평균값: {attention_matrix.mean():.6f}")
    print(f"표준편차: {attention_matrix.std():.6f}")
    print(f"중앙값: {np.median(attention_matrix):.6f}")
    
    # 0이 아닌 값들만 분석
    non_zero_values = attention_matrix[attention_matrix > 0]
    if len(non_zero_values) > 0:
        print(f"0이 아닌 값들의 개수: {len(non_zero_values)}")
        print(f"0이 아닌 값들의 최솟값: {non_zero_values.min():.6f}")
        print(f"0이 아닌 값들의 최댓값: {non_zero_values.max():.6f}")
    
    # 원본 값 저장 (나중에 참고용)
    original_attention_matrix = attention_matrix.copy()
    
    # 정규화
    if attention_matrix.max() != attention_matrix.min():
        attention_matrix = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min())
    
    print(f"\n정규화 후:")
    print(f"최솟값: {attention_matrix.min():.6f}")
    print(f"최댓값: {attention_matrix.max():.6f}")
    print(f"평균값: {attention_matrix.mean():.6f}")
    
    # 상위 15개 문제 선택 (빈도 기준)
    top_problems = sorted(problem_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    top_problem_indices = [prob[0] for prob in top_problems]
    
    print(f"\n상위 15개 문제 (빈도 기준):")
    for i, (prob_idx, freq) in enumerate(top_problems):
        idx2q = {idx: q for q, idx in dataset.q2idx.items()}
        problem_name = idx2q[prob_idx]
        print(f"{i+1}. {problem_name} (빈도: {freq})")
    
    # 상위 15개 문제에 대한 attention matrix 추출
    top_attention_matrix = attention_matrix[np.ix_(top_problem_indices, top_problem_indices)]
    
    # SLAM 전용 문제 번역 사전 (듀오링고 학습 방식)
    problem_translations = {
        'is_reverse_translate': '문장 역번역',
        'I_reverse_translate': '인칭대명사 역번역',
        'I_listen': '인칭대명사 듣기',
        'The_reverse_translate': '정관사 역번역',
        'the_reverse_translate': '정관사 역번역',  # 소문자 버전 추가
        'I_reverse_tap': '인칭대명사 단어선택',
        'my_reverse_translate': '소유대명사 역번역',
        'is_reverse_tap': '동사 단어선택',
        'the_reverse_tap': '정관사 단어선택',  # 추가된 번역
        'The_reverse_tap': '정관사 단어선택',  # 대문자 버전 추가
        'is_listen': '동사 듣기',  # 추가된 번역
        'The_listen': '정관사 듣기',
        'a_reverse_translate': '부정관사 역번역',
        'and_reverse_translate': '접속사 역번역',
        'are_reverse_translate': '복수동사 역번역',
        'my_listen': '소유대명사 듣기',
        'reverse_translate': '일반 역번역',
        'reverse_tap': '단어선택 번역',
        'listen': '듣기 연습',
        # 추가 SLAM 패턴들
        'you_reverse_translate': '2인칭 역번역',
        'you_listen': '2인칭 듣기',
        'you_reverse_tap': '2인칭 단어선택',
        'we_reverse_translate': '1인칭복수 역번역',
        'we_listen': '1인칭복수 듣기',
        'we_reverse_tap': '1인칭복수 단어선택',
        'they_reverse_translate': '3인칭복수 역번역',
        'they_listen': '3인칭복수 듣기',
        'they_reverse_tap': '3인칭복수 단어선택',
        'have_reverse_translate': '완료시제 역번역',
        'have_listen': '완료시제 듣기',
        'have_reverse_tap': '완료시제 단어선택',
        'will_reverse_translate': '미래시제 역번역',
        'will_listen': '미래시제 듣기',
        'will_reverse_tap': '미래시제 단어선택'
    }
    
    # 선택된 문제들에 대한 레이블 생성
    labels = []
    idx2q = {idx: q for q, idx in dataset.q2idx.items()}
    for idx in top_problem_indices:
        name = idx2q[idx]
        kor_name = problem_translations.get(name.split('(')[0].strip(), name)
        if idx in problem_correct_count:
            stats = problem_correct_count[idx]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            freq = problem_freq[idx]
            label = f"{kor_name}\n(출현: {freq}회, 정답률: {accuracy:.0%})"
        else:
            label = f"{kor_name}\n(데이터 없음)"
        labels.append(label)
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 히트맵 생성
    plt.figure(figsize=(15, 12))
    
    # SLAM 전용 색상 스케일 (0~1 고정)
    sns.heatmap(top_attention_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Attention 가중치 (정규화됨)'},
                square=True,
                vmin=0,  # 최솟값 고정
                vmax=1)  # 최댓값 고정 (SLAM 전용)
    
    plt.title('SLAM 데이터셋: 전체 학생 데이터 기반 문제 간 영향력 분석', pad=20, fontsize=16)
    plt.xlabel('이전에 푼 문제', labelpad=10)
    plt.ylabel('현재 풀고 있는 문제', labelpad=10)
    
    # X축 라벨 대각선으로 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    fig = plt.gcf()
    
    if return_matrix:
        return fig, attention_matrix, original_attention_matrix, labels
    else:
        return fig
