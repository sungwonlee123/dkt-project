import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import inspect

def visualize_global_attention_patterns(model, dataset, sequence_length=100, return_matrix=False):
    """
    전체 데이터셋에 대한 Attention 패턴을 시각화하는 함수
    Args:
        model: 학습된 SAKT 모델
        dataset: 데이터셋
        sequence_length: 시각화할 시퀀스 길이 (기본값: 10)
    """
    print("\nAnalyzing global attention patterns...")
    
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
        
        # 디버깅 출력 (필요시 주석 해제)
        # print(f"DEBUG: Dataset attributes - d_seqs: {has_diff}, s_seqs: {has_static_diff}, a_seqs: {has_attempts}, t_seqs: {has_time}")
        
        d_seq = dataset.d_seqs[i] if has_diff else np.zeros_like(q_seq)
        s_seq = dataset.s_seqs[i] if has_static_diff else d_seq
        a_seq = dataset.a_seqs[i] if has_attempts else np.zeros_like(q_seq)
        t_seq = dataset.t_seqs[i] if has_time else np.zeros_like(q_seq)
        # 패딩된 부분 제거
        valid_indices = q_seq != pad_val
        seq_len = valid_indices.sum()
        # 모든 feature 시퀀스 길이 맞추기 (불일치 시 최소 길이로 자름)
        min_len = min(len(q_seq), len(r_seq), len(d_seq), len(s_seq), len(a_seq), len(t_seq), len(valid_indices))
        q_seq = q_seq[:min_len][valid_indices[:min_len]]
        r_seq = r_seq[:min_len][valid_indices[:min_len]]
        d_seq = d_seq[:min_len][valid_indices[:min_len]]
        s_seq = s_seq[:min_len][valid_indices[:min_len]]
        a_seq = a_seq[:min_len][valid_indices[:min_len]]
        t_seq = t_seq[:min_len][valid_indices[:min_len]]
        # If any sequence is shorter than window_size, pad it
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
        
        if len(q_seq) < 2:  # 최소 2개의 문제가 필요
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
        
        # Attention 패턴 계산
        with torch.no_grad():
            # 시퀀스를 분석 가능한 크기로 자르기
            if len(q_seq) < 2:  # 최소 2개의 문제가 필요
                continue
                
            # 시퀀스가 너무 짧으면 건너뛰기
            if len(q_seq) < 2:
                continue
            
            # 시퀀스를 작은 윈도우로 처리
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
            
            # 입력 텐서 준비 및 타입 변환
            q = q.long()
            r = r.long()
            
            # 모델의 forward 메소드 시그니처를 동적으로 확인하여 적절한 파라미터로 호출
            try:
                # forward 메소드의 파라미터 확인
                forward_signature = inspect.signature(model.forward)
                param_names = list(forward_signature.parameters.keys())
                
                # print(f"DEBUG: Model parameters: {param_names}, count: {len(param_names)}")
                
                # 파라미터 개수에 따라 적절히 호출 (복잡한 모델부터 체크)
                if len(param_names) >= 7:  # q, r, qry, d_diff, s_diff, att, time
                    # DKT2 (다중 특성) - 7개 파라미터
                    # print("DEBUG: Using DKT2 7-parameter call")
                    # None 체크 및 기본값 할당
                    d = d if d is not None else torch.zeros_like(q, dtype=torch.float32)
                    s = s if s is not None else torch.zeros_like(q, dtype=torch.float32)
                    a = a if a is not None else torch.zeros_like(q, dtype=torch.float32)
                    t = t if t is not None else torch.zeros_like(q, dtype=torch.float32)
                    _, curr_attention = model(q, r, q, d, s, a, t)
                elif len(param_names) >= 6:  # q, r, qry, d, a, t
                    # DKT3 - 6개 파라미터
                    # print("DEBUG: Using DKT3 6-parameter call")
                    _, curr_attention = model(q, r, q, d, a, t)
                elif 'd_diff' in param_names and len(param_names) == 4:
                    # SLAM with difficulty (d_diff 파라미터 있음) - 4개 파라미터
                    # print("DEBUG: Using SLAM 4-parameter call")
                    _, curr_attention = model(q, r, q, d)
                else:
                    # SAKT/DKT-old (기본 모델: q, r, qry) - 3개 파라미터
                    # print("DEBUG: Using basic 3-parameter call")
                    _, curr_attention = model(q, r, q)
            except (TypeError, RuntimeError) as e:
                # DKT2의 경우 에러 로그를 간소화 (너무 많은 출력 방지)
                if len(param_names) >= 7:
                    pass  # DKT2는 에러 출력 생략
                else:
                    print(f"모델 호출 에러: {e}")
                    print("기본 파라미터로 재시도합니다.")
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
    
    # 통계 출력
    total_interactions = sum(problem_freq.values())
    total_students = len(set(dataset.u_list))
    print(f"\n전체 통계:")
    print(f"총 학생 수: {total_students}명")
    print(f"총 상호작용 수: {total_interactions}회")
    print(f"학생당 평균 문제 풀이 수: {total_interactions/total_students:.1f}회")
    
    # 가장 많이 등장하는 상위 15개 문제 유형 선택
    top_problems = sorted(problem_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    top_problem_indices = [idx for idx, _ in top_problems]
    
    print("\n상위 15개 문제 통계:")
    for idx, freq in top_problems:
        correct = problem_correct_count[idx]['correct']
        total = problem_correct_count[idx]['total']
        accuracy = correct / total if total > 0 else 0
        print(f"문제 {idx}: {total}회 출현 (전체의 {total/total_interactions*100:.1f}%), 정답률 {accuracy*100:.1f}%")
    
    # 선택된 문제들에 대한 attention matrix 추출
    attention_sum_subset = problem_attention_sum[top_problem_indices][:, top_problem_indices]
    attention_count_subset = problem_attention_count[top_problem_indices][:, top_problem_indices]
    
    attention_matrix = np.divide(
        attention_sum_subset,
        attention_count_subset,
        out=np.zeros((len(top_problem_indices), len(top_problem_indices))),
        where=attention_count_subset!=0
    )
    
    # 정규화 전 실제 값 확인
    print(f"\n정규화 전 실제 어텐션 값 분석:")
    print(f"최솟값: {attention_matrix.min():.6f}")
    print(f"최댓값: {attention_matrix.max():.6f}")
    print(f"평균값: {attention_matrix.mean():.6f}")
    print(f"표준편차: {attention_matrix.std():.6f}")
    print(f"중앙값: {np.median(attention_matrix):.6f}")
    
    # 실제 값 분포 확인
    non_zero_values = attention_matrix[attention_matrix > 0]
    if len(non_zero_values) > 0:
        print(f"0이 아닌 값들의 평균: {non_zero_values.mean():.6f}")
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
    
    # 문제 유형별 정답률 계산
    for stats in problem_stats.values():
        stats['accuracy'] = stats['correct'] / stats['total']
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Figure: 히트맵
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
        'Ordering Positive Decimals': '양의 소수 크기 비교',
        'Equivalent Fractions': '동치분수',
        'Circle Graph': '원그래프',
        'Proportion': '비례',
        'Finding Percents': '백분율',
        'Pythagorean Theorem': '피타고라스 정리',
        'Box and Whisker': '상자 그림',
        'Histogram': '히스토그램',
        'Mean': '평균',
        'Median': '중앙값',
        'Mode': '최빈값',
        'Range': '범위',
        'Stem and Leaf Plot': '줄기와 잎 그림',
        'Pattern Finding': '패턴 찾기',
        'Scientific Notation': '지수 표현',
        'Area': '넓이',
        'Volume': '부피',
        'Surface Area': '표면적',
        'Order of Operations': '연산 순서',
        'Solving for a Variable': '변수 풀이',
        'Greatest Common Factor': '최대공약수',
        'Least Common Multiple': '최소공배수',
        'Ratios': '비율',
        'Multiplication': '곱셈',
        'Division': '나눗셈',
        'Addition': '덧셈',
        # SLAM 데이터용 추가 (듀오링고 학습 방식)
        'is_reverse_translate': '문장 역번역',
        'I_reverse_translate': '인칭대명사 역번역',
        'I_listen': '인칭대명사 듣기',
        'The_reverse_translate': '정관사 역번역',
        'the_reverse_translate': '정관사 역번역',  # 소문자 버전 추가
        'I_reverse_tap': '인칭대명사 단어선택',
        'my_reverse_translate': '소유대명사 역번역',
        'is_reverse_tap': '동사 단어선택',
        'the_reverse_tap': '정관사 단어선택',  # 소문자 버전
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
    
    # 히트맵 그리기 - 더 정밀한 표시를 위해 fmt 변경
    sns.heatmap(attention_matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',  # 소수점 3자리까지 표시
                square=True,
                cbar_kws={'label': '영향력 점수'},
                ax=ax)
    
    ax.set_title('전체 학생 데이터 기반 문제 간 영향력 분석', pad=20, fontsize=14)
    ax.set_xlabel('이전에 푼 문제', labelpad=10)
    ax.set_ylabel('현재 풀고 있는 문제', labelpad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # 범례 추가
    legend_text = """
    < 히트맵 해석 방법 >
    
    1. 데이터 의미
       ∙ 세로축: 현재 풀고 있는 문제
       ∙ 가로축: 이전에 푼 문제들
       ∙ 칸의 값: 이전 문제가 현재 문제에 미치는 영향력 (0~1)
       ∙ 괄호 안 정보: (전체 출현 횟수, 전체 정답률)
    
    2. 색상 의미
       ∙ 색이 진할수록 영향력이 큼
       ∙ 흰색: 영향력 없음 (0.0)
       ∙ 진한 빨강: 매우 높은 영향력 (1.0)
    
    3. 활용 방법
       ∙ 진한 색의 연결: 강한 연관성이 있는 문제 쌍
       ∙ 흰색의 연결: 서로 독립적인 문제 쌍
       ∙ 대각선: 같은 유형의 문제를 연속해서 풀 때의 영향력
    
    * 이 분석은 전체 학생들(총 {num_students}명)의 모든 문제 풀이 데이터를 
      기반으로 계산된 평균적인 영향력을 보여줍니다.
    """.format(num_students=dataset.num_u)
    plt.figtext(1.02, 0.5, legend_text, fontsize=10, va='center')
    
    # 그래프 크기와 여백 조정
    plt.gcf().set_size_inches(15, 12)
    plt.tight_layout()
    
    # 범례 위치 조정
    plt.figtext(1.02, 0.5, legend_text, 
                fontsize=11, 
                va='center',
                bbox=dict(facecolor='white', 
                         edgecolor='lightgray',
                         boxstyle='round,pad=1'))
    
    if return_matrix:
        return fig, attention_matrix, original_attention_matrix, labels
    else:
        return fig
