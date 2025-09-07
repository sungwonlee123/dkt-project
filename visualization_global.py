import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def visualize_global_attention_patterns(model, dataset, sequence_length=100):
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
    for i in tqdm(range(len(dataset.q_seqs)), desc="Processing sequences"):
        # 시퀀스 준비 - 원래 길이 유지
        q_seq = dataset.q_seqs[i]
        r_seq = dataset.r_seqs[i]
        
        # 추가 feature가 있는 경우에만 가져오기
        has_diff = hasattr(dataset, 'd_seqs')
        has_static_diff = hasattr(dataset, 's_seqs')
        has_attempts = hasattr(dataset, 'a_seqs')
        has_time = hasattr(dataset, 't_seqs')
        
        d_seq = dataset.d_seqs[i] if has_diff else np.zeros_like(q_seq)
        s_seq = dataset.s_seqs[i] if has_static_diff else d_seq  # 정적 난이도가 없으면 동적 난이도 사용
        a_seq = dataset.a_seqs[i] if has_attempts else np.zeros_like(q_seq)
        t_seq = dataset.t_seqs[i] if has_time else np.zeros_like(q_seq)
        
        # 패딩된 부분 제거
        valid_indices = q_seq != pad_val
        q_seq = q_seq[valid_indices]
        r_seq = r_seq[valid_indices]
        d_seq = d_seq[valid_indices]
        s_seq = s_seq[valid_indices]
        a_seq = a_seq[valid_indices]
        t_seq = t_seq[valid_indices]
        
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
            
            try:
                # DKT2 모델의 경우
                if hasattr(model, 'feature_projection'):  # DKT2 모델의 특징
                    _, curr_attention = model(q, r, q, d, s, a, t)
                # DKT3 모델의 경우
                elif has_diff and has_attempts and has_time:
                    _, curr_attention = model(q, r, q, d, a, t)
                # 기본 SAKT 모델(DKT-old)의 경우
                else:
                    _, curr_attention = model(q, r, q)
            except TypeError as e:
                print(f"모델 호출 에러: {e}, 기본 파라미터로 시도합니다.")
                # 항상 동작하는 기본 호출
                _, curr_attention = model(q, r, q)
            
            curr_attention = curr_attention[0].cpu().numpy() 
            
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
    attention_matrix = np.divide(
        problem_attention_sum[top_problem_indices][:, top_problem_indices],
        problem_attention_count[top_problem_indices][:, top_problem_indices],
        out=np.zeros((len(top_problem_indices), len(top_problem_indices))),
        where=problem_attention_count[top_problem_indices][:, top_problem_indices]!=0
    )
    
    # 정규화
    if attention_matrix.max() != attention_matrix.min():
        attention_matrix = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min())
    
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
    
    # 히트맵 그리기
    sns.heatmap(attention_matrix, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
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
    
    return fig
