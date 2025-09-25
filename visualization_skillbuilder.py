import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import inspect

def visualize_skillbuilder_attention_patterns(model, dataset, sequence_length=100, return_matrix=False, model_name='skillbuilder'):
    """
    Skill Builder 데이터셋 전용 Attention 패턴 시각화 함수 (DKT Old, DKT2)
    Args:
        model: 학습된 SAKT 모델
        dataset: 데이터셋
        sequence_length: 시각화할 시퀀스 길이 (기본값: 100)
        return_matrix: 매트릭스 반환 여부
    """
    print("\nAnalyzing Skill Builder attention patterns...")
    
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
    
    # 원래 시퀀스 제한 방식으로 복원
    min_seq_len = min(seq_lens)
    print(f"처리할 시퀀스 개수: {min_seq_len} (원래 방식)")
    for i in tqdm(range(min_seq_len), desc="Processing Skill Builder sequences"):
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
        
        # 시퀀스 길이 조정 (더 많은 정보 활용)
        window_size = min(50, seq_len)  # 50으로 증가 (더 긴 어텐션 패턴)
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
            
        # 문제 유형별 정답/오답 카운트 (안전한 처리)
        for q, r in zip(q_seq, r_seq):
            if q != pad_val and q < dataset.num_q:  # 유효한 문제 번호인지 확인
                problem_name = f"Problem_{q}"  # 간단한 문제명 생성
                if problem_name not in problem_stats:
                    problem_stats[problem_name] = {'correct': 0, 'total': 0}
                problem_stats[problem_name]['total'] += 1
                if r == 1:
                    problem_stats[problem_name]['correct'] += 1
        
        # 시퀀스를 윈도우로 처리 (이미 위에서 설정됨)
        if len(q_seq) < 2:  # 최소 2개 문제는 있어야 어텐션 계산 가능
            continue
            
        window_q = q_seq[:window_size]
        window_r = r_seq[:window_size]
        window_d = d_seq[:window_size]
        window_s = s_seq[:window_size]
        window_a = a_seq[:window_size]
        window_t = t_seq[:window_size]
        
        # 패딩 처리 (모델 입력 크기에 맞춤)
        if hasattr(model, 'n') and model.n > window_size:
            padding_length = model.n - window_size
            q_pad = np.pad(window_q, (0, padding_length), constant_values=0)
            r_pad = np.pad(window_r, (0, padding_length), constant_values=0)
            d_pad = np.pad(window_d, (0, padding_length), constant_values=0.0)
            s_pad = np.pad(window_s, (0, padding_length), constant_values=0.0)
            a_pad = np.pad(window_a, (0, padding_length), constant_values=0.0)
            t_pad = np.pad(window_t, (0, padding_length), constant_values=0.0)
        else:
            # 패딩 없이 사용
            q_pad, r_pad = window_q, window_r
            d_pad, s_pad = window_d, window_s
            a_pad, t_pad = window_a, window_t
        
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
        
        # 모델의 forward 메소드 시그니처를 동적으로 확인하여 적절한 파라미터로 호출
        # 완전히 통일된 시각화: 모든 모델을 동일하게 처리
        # 모델 구분 없이 3개 파라미터로만 호출
        _, curr_attention = model(q, r, q)
        
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
    
    # 평균 계산 (0인 경우 실제로 0으로 유지)
    attention_matrix = np.zeros_like(problem_attention_sum)
    valid_mask = problem_attention_count > 0
    attention_matrix[valid_mask] = problem_attention_sum[valid_mask] / problem_attention_count[valid_mask]
    
    # 문제 빈도 계산 (실제 출현 빈도)
    problem_freq = {}
    for i in range(dataset.num_q):
        # 실제 문제 출현 빈도 계산
        freq = 0
        for j in range(len(dataset.q_seqs)):
            if j < len(dataset.q_seqs):
                freq += np.sum(np.array(dataset.q_seqs[j]) == i)
        problem_freq[i] = freq
    
    print(f"\nSkill Builder 데이터셋 어텐션 값 분석:")
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
        if hasattr(dataset, 'q2idx') and dataset.q2idx:
            try:
                idx2q = {idx: q for q, idx in dataset.q2idx.items()}
                problem_name = idx2q.get(prob_idx, f"Problem_{prob_idx}")
            except:
                problem_name = f"Problem_{prob_idx}"
        else:
            problem_name = f"Problem_{prob_idx}"
        print(f"{i+1}. {problem_name} (빈도: {freq})")
    
    # 상위 15개 문제에 대한 attention matrix 추출
    top_original_attention_matrix = original_attention_matrix[np.ix_(top_problem_indices, top_problem_indices)]
    top_attention_matrix = attention_matrix[np.ix_(top_problem_indices, top_problem_indices)]
    
    # Basic: 상위 15개 매트릭스를 다시 정규화 (0~1)
    if top_attention_matrix.max() != top_attention_matrix.min():
        top_attention_matrix = (top_attention_matrix - top_attention_matrix.min()) / (top_attention_matrix.max() - top_attention_matrix.min())
    
    # Skill Builder 전용 문제 번역 사전 (확장)
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
        'Ordering Positive Decimals': '양의 소수 크기 비교',
        'Equivalent Fractions': '동치분수',
        'Circle Graph': '원그래프',
        'Proportion': '비례',
        'Finding Percents': '백분율 구하기',
        'Pythagorean Theorem': '피타고라스 정리',
        'Box and Whisker': '상자 그림',
        'Histogram': '히스토그램',
        # 추가 번역 (히트맵에서 확인된 미번역 문제들)
        'Ordering Fractions': '분수의 크기 비교',
        'Multiplication and Division Integers': '정수의 곱셈과 나눗셈',
        'Addition and Subtraction Positive Decimals': '양의 소수의 덧셈과 뺄셈',
        'Table': '표 읽기',
        'Median': '중앙값',
        'Addition Whole Numbers': '자연수의 덧셈',
        'Ordering Integers': '정수의 크기 비교',
        'Mean': '평균',
        'Mode': '최빈값',
        'Range': '범위',
        'Stem and Leaf Plot': '줄기-잎 그림',
        'Bar Graph': '막대그래프',
        'Line Graph': '선그래프',
        'Pictograph': '그림그래프',
        'Multiplication Whole Numbers': '자연수의 곱셈',
        'Division Whole Numbers': '자연수의 나눗셈',
        'Rounding': '반올림',
        'Estimation': '어림하기',
        'Place Value': '자릿값',
        'Prime Factorization': '소인수분해',
        'GCD and LCM': '최대공약수와 최소공배수',
        'Ratio': '비',
        'Rate': '비율',
        'Unit Rate': '단위비율',
        'Scale Factor': '축척',
        'Similar Figures': '닮은 도형',
        'Congruent Figures': '합동 도형',
        'Area': '넓이',
        'Perimeter': '둘레',
        'Volume': '부피',
        'Surface Area': '겉넓이',
        'Pattern Finding': '패턴 찾기',
        'Scientific Notation': '지수 표현',
        'Order of Operations': '연산 순서',
        'Solving for a Variable': '변수 풀이',
        'Greatest Common Factor': '최대공약수',
        'Least Common Multiple': '최소공배수',
        'Ratios': '비율',
        'Multiplication': '곱셈',
        'Division': '나눗셈',
        'Addition': '덧셈',
        'Subtraction': '뺄셈',
        'Decimals': '소수',
        'Fractions': '분수',
        'Mixed Numbers': '대분수',
        'Linear Equations': '일차방정식',
        'Inequalities': '부등식',
        'Geometry': '기하',
        'Angles': '각도',
        'Coordinates': '좌표',
        'Data Analysis': '자료 분석',
        'Probability': '확률',
        'Statistics': '통계',
        'Word Problems': '문장제'
    }
    
    # 선택된 문제들에 대한 레이블 생성 (안전한 처리)
    labels = []
    for idx in top_problem_indices:
        # q2idx가 없는 경우 간단한 문제명 사용
        if hasattr(dataset, 'q2idx') and dataset.q2idx:
            try:
                idx2q = {idx: q for q, idx in dataset.q2idx.items()}
                name = idx2q.get(idx, f"Problem_{idx}")
                kor_name = problem_translations.get(name.split('(')[0].strip(), name)
            except:
                kor_name = f"문제_{idx}"
        else:
            kor_name = f"문제_{idx}"
            
        if idx in problem_correct_count:
            stats = problem_correct_count[idx]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            freq = problem_freq[idx]
            label = f"{kor_name}\n(출현: {freq}회, 정답률: {accuracy:.0%})"
        else:
            label = f"{kor_name}\n(데이터 없음)"
        labels.append(label)
    
    # 한글 폰트 설정 (macOS 전용)
    plt.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 히트맵 생성
    plt.figure(figsize=(15, 12))
    
    # Skill Builder 전용 색상 스케일 (0~1 고정)
    sns.heatmap(top_attention_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Attention 가중치 (정규화됨)'},
                square=True,
                vmin=0,  # 최솟값 고정
                vmax=1)  # 최댓값 고정 (Skill Builder 전용)
    
    plt.title('Skill Builder 데이터셋: 전체 학생 데이터 기반 문제 간 영향력 분석', pad=20, fontsize=16)
    plt.xlabel('이전에 푼 문제', labelpad=10)
    plt.ylabel('현재 풀고 있는 문제', labelpad=10)
    
    # X축 라벨 대각선으로 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    fig = plt.gcf()
    
    # 정규화 전 히트맵도 저장
    plt.figure(figsize=(15, 12))
    
    # 정규화 전 원본 히트맵 (실제 어텐션 값)
    sns.heatmap(top_original_attention_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Attention 가중치 (정규화 전)'},
                square=True)
    
    plt.title('Skill Builder 데이터셋: 정규화 전 원본 어텐션 히트맵', pad=20, fontsize=16)
    plt.xlabel('이전에 푼 문제', labelpad=10)
    plt.ylabel('현재 풀고 있는 문제', labelpad=10)
    
    # X축 라벨 대각선으로 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # 정규화 전 히트맵 저장 (모델별 구분)
    # model_name은 함수 매개변수로 전달받음
    
    plt.savefig(f'attention_heatmap_global_{model_name}_nonnormalized.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Non-normalized global attention visualization saved")
    
    if return_matrix:
        # Enhanced 시각화를 위해 정규화된 매트릭스와 원본 매트릭스 모두 반환
        return fig, top_attention_matrix, top_original_attention_matrix, labels
    else:
        return fig
