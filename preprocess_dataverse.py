import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

def read_slam_data(file_path):
    """
    SLAM 데이터를 읽어서 DataFrame으로 변환
    """
    data = []
    current_user = None
    current_days = None
    current_format = None
    current_time = None
    record_count = 0
    print("Reading file:", file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.startswith('# prompt:'):
                    continue
                elif line.startswith('# user:'):
                    # 새로운 사용자 데이터 시작
                    line = line.strip('# ')
                    user_info = {}
                    for part in line.split('  '):
                        if ':' in part:
                            key, value = part.split(':', 1)
                            user_info[key] = value
                    
                    if 'user' in user_info:
                        current_user = user_info['user']
                        current_days = float(user_info.get('days', '0'))
                        current_format = user_info.get('format', 'unknown')
                        time_str = user_info.get('time', '0').strip().lower()
                        current_time = 0 if time_str == 'null' else int(time_str)
                        print(f"User info: {user_info}")
                elif not line.startswith('#'):
                    # 토큰 데이터 라인
                    parts = line.strip().split()
                    if len(parts) >= 3:  # 최소한 토큰과 position, correct가 있어야 함
                        # 마지막 두 필드는 position과 correct
                        correct = int(parts[-1])    # 마지막 필드는 correct (0 또는 1)
                        position = int(parts[-2])   # 마지막에서 두 번째 필드는 position
                        
                        # 토큰과 나머지 정보
                        token = parts[1]  # 두 번째 필드는 항상 토큰
                        
                        # 중간의 형태소 정보들은 무시하고 필요한 정보만 저장
                        pos = parts[2] if len(parts) > 2 else "UNK"  # 품사 정보 (있는 경우만)
                            
                        data.append({
                            'user': current_user,
                            'days': current_days,
                            'time': current_time,
                            'token': token,
                            'part_of_speech': pos,
                            'token_position': position,
                            'correct': correct,
                            'format': current_format
                        })
            except Exception as e:
                print(f"Error at line {line_num}: {line.strip()}")
                print(f"Error details: {str(e)}")
                continue

        print(f"Total records read: {len(data)}")
    
    return pd.DataFrame(data)

def preprocess_dataverse():
    """
    Dataverse SLAM 데이터를 DKT 형식으로 전처리
    """
    print("\n=== Dataverse SLAM 데이터 전처리 시작 ===")
    
    # 영어-스페인어 데이터 로드
    print("\n1. 데이터 로드 중...")
    train_path = '/Users/iseong-won/dkt project/dataverse_files/data_en_es/en_es.slam.20190204.train'
    df = read_slam_data(train_path)
    
    print("\n원본 데이터 정보:")
    print(df.info())
    print("\n샘플 데이터:")
    print(df.head())
    
    # 문제 ID 생성 (token + format 조합)
    print("\n2. 문제 ID 생성 중...")
    df['item_id'] = df['token'] + '_' + df['format']
    
    # 문제별 통계 계산
    print("\n3. 문제별 통계 계산 중...")
    problem_stats = df.groupby('item_id').agg({
        'correct': ['count', 'mean']
    }).reset_index()
    problem_stats.columns = ['item_id', 'attempts', 'correct_rate']
    
    # 난이도 계산 (1 - 정답률)
    problem_stats['difficulty'] = 1 - problem_stats['correct_rate']
    
    # Z-score 정규화
    problem_stats['difficulty'] = (problem_stats['difficulty'] - problem_stats['difficulty'].mean()) / problem_stats['difficulty'].std()
    
    # 데이터프레임 병합
    df = df.merge(problem_stats[['item_id', 'difficulty']], on='item_id')
    
    # 시도 횟수와 누적 정답률 계산
    print("\n4. 학습자별 통계 계산 중...")
    df = df.sort_values(['user', 'time'])
    df['cumulative_attempts'] = df.groupby(['user', 'item_id']).cumcount() + 1
    df['running_correct'] = df.groupby(['user', 'item_id'])['correct'].cumsum()
    df['running_accuracy'] = df['running_correct'] / df['cumulative_attempts']
    
    # 최종 데이터프레임 구성
    print("\n5. 최종 데이터 구성 중...")
    final_df = df[[
        'user',
        'item_id',
        'correct',
        'difficulty',
        'cumulative_attempts',
        'running_accuracy',
        'time',
        'days'
    ]].rename(columns={
        'user': 'user_id',
        'days': 'days_in_course'
    })
    
    # 데이터 저장
    print("\n6. 전처리된 데이터 저장 중...")
    final_df.to_csv('processed_dataverse_data.csv', index=False)
    
    # 데이터 통계 출력
    print("\n=== 전처리 완료 ===")
    print(f"총 데이터 수: {len(final_df):,}")
    print(f"고유 사용자 수: {final_df['user_id'].nunique():,}")
    print(f"고유 문제 수: {final_df['item_id'].nunique():,}")
    print(f"평균 정답률: {final_df['correct'].mean():.2%}")
    
    # 메타데이터 저장
    metadata = {
        'total_records': len(final_df),
        'unique_users': final_df['user_id'].nunique(),
        'unique_items': final_df['item_id'].nunique(),
        'avg_correct_rate': float(final_df['correct'].mean()),
        'features': {
            'user_id': '학습자 식별자',
            'item_id': '문제 식별자 (token_format)',
            'correct': '정답 여부 (0 또는 1)',
            'difficulty': '문제 난이도 (Z-score)',
            'cumulative_attempts': '누적 시도 횟수',
            'running_accuracy': '현재까지의 정답률',
            'time': '타임스탬프',
            'days_in_course': '과정 진행 일수'
        }
    }
    
    with open('dataverse_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("\n메타데이터가 'dataverse_metadata.json'에 저장되었습니다.")
    
    return final_df

if __name__ == "__main__":
    # 데이터 전처리 실행
    df = preprocess_dataverse()
