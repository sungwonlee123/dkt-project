#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assistment 수학 데이터셋 전처리 스크립트 (Model 1)
수학 개념간 유사도를 측정하여 데이터셋을 향상시킵니다.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
warnings.filterwarnings('ignore')

def extract_math_keywords(skill_name):
    """
    수학 스킬 이름에서 핵심 키워드를 추출합니다.
    """
    if pd.isna(skill_name):
        return []
    
    # 수학 관련 키워드 매핑
    math_keywords = {
        # 기본 연산
        'addition': ['add', 'plus', 'sum', 'total'],
        'subtraction': ['subtract', 'minus', 'difference', 'take away'],
        'multiplication': ['multiply', 'times', 'product'],
        'division': ['divide', 'quotient', 'split'],
        
        # 분수 관련
        'fraction': ['fraction', 'numerator', 'denominator', 'ratio'],
        'decimal': ['decimal', 'point', 'tenth', 'hundredth'],
        'percent': ['percent', 'percentage', '%'],
        
        # 기하학
        'geometry': ['area', 'perimeter', 'volume', 'surface', 'angle', 'triangle', 'rectangle', 'circle'],
        'measurement': ['length', 'width', 'height', 'radius', 'diameter'],
        
        # 대수학
        'algebra': ['equation', 'variable', 'solve', 'expression', 'formula'],
        'graph': ['graph', 'plot', 'coordinate', 'axis'],
        
        # 통계
        'statistics': ['mean', 'median', 'mode', 'average', 'range', 'probability'],
        'data': ['data', 'table', 'chart', 'diagram', 'venn'],
        
        # 기타
        'number': ['integer', 'whole', 'positive', 'negative', 'prime', 'factor'],
        'pattern': ['pattern', 'sequence', 'rule']
    }
    
    skill_lower = skill_name.lower()
    extracted_keywords = []
    
    for category, keywords in math_keywords.items():
        for keyword in keywords:
            if keyword in skill_lower:
                extracted_keywords.append(category)
                break
    
    return extracted_keywords

def calculate_skill_similarity(skills_list):
    """
    수학 스킬들 간의 유사도를 계산합니다.
    """
    print("🔍 수학 개념 유사도 계산 중...")
    
    # 1. 키워드 추출
    skill_keywords = {}
    for skill in skills_list:
        if pd.notna(skill):
            skill_keywords[skill] = extract_math_keywords(skill)
    
    # 2. TF-IDF 벡터화를 위한 텍스트 생성
    skill_texts = []
    for skill, keywords in skill_keywords.items():
        if keywords:
            skill_texts.append(' '.join(keywords))
        else:
            skill_texts.append(skill.lower())  # 키워드가 없으면 원본 사용
    
    # 3. TF-IDF 벡터화
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(skill_texts)
    
    # 4. 코사인 유사도 계산
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # 5. 유사도 결과를 DataFrame으로 변환
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=skills_list,
        columns=skills_list
    )
    
    return similarity_df, skill_keywords

def find_similar_skills(similarity_df, threshold=0.3):
    """
    유사도가 높은 스킬 쌍을 찾습니다.
    """
    similar_pairs = []
    
    for i in range(len(similarity_df)):
        for j in range(i+1, len(similarity_df)):
            skill1 = similarity_df.index[i]
            skill2 = similarity_df.columns[j]
            similarity = similarity_df.iloc[i, j]
            
            if similarity > threshold:
                similar_pairs.append({
                    'skill1': skill1,
                    'skill2': skill2,
                    'similarity': similarity
                })
    
    return pd.DataFrame(similar_pairs).sort_values('similarity', ascending=False)

def preprocess_assistment_model1(input_file='skill_builder_model1.csv', output_file='skill_builder_model1_processed.csv'):
    """
    Assistment 데이터셋을 수학 개념 유사도와 함께 전처리합니다.
    
    Args:
        input_file (str): 입력 CSV 파일 경로
        output_file (str): 출력 CSV 파일 경로
    """
    
    print("🚀 Assistment 수학 데이터셋 전처리 시작 (Model 1)")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("📁 데이터 로딩 중...")
    try:
        df = pd.read_csv(input_file, encoding='latin-1', low_memory=False)
    except:
        df = pd.read_csv(input_file, encoding='utf-8', low_memory=False)
    
    print(f"원본 데이터 크기: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    
    # 2. 기본 전처리
    print("\n🔧 기본 전처리 중...")
    
    # skill_name이 있는 데이터만 선택 (NA 제거)
    df_clean = df.dropna(subset=['skill_name'])
    print(f"skill_name NA 제거 후: {df_clean.shape} (제거된 행: {len(df) - len(df_clean):,}개)")
    
    # 중복 제거
    df_clean = df_clean.drop_duplicates()
    print(f"중복 제거 후: {df_clean.shape} (제거된 행: {len(df) - len(df_clean):,}개)")
    
    # 3. 수학 개념 유사도 분석
    print("\n📊 수학 개념 유사도 분석 중...")
    
    # 고유한 스킬 목록 추출
    unique_skills = df_clean['skill_name'].dropna().unique()
    print(f"고유한 수학 스킬 수: {len(unique_skills)}")
    
    # 스킬별 출현 빈도 확인
    skill_counts = df_clean['skill_name'].value_counts()
    print(f"스킬별 출현 빈도 통계:")
    print(f"  최대: {skill_counts.max():,}회")
    print(f"  최소: {skill_counts.min():,}회")
    print(f"  평균: {skill_counts.mean():.1f}회")
    
    # 전체 스킬로 분석 (112개 스킬 모두 포함)
    print(f"전체 {len(unique_skills)}개 스킬로 유사도 분석 시작...")
    
    # 유사도 계산 (전체 스킬)
    similarity_df, skill_keywords = calculate_skill_similarity(unique_skills)
    
    # 유사한 스킬 쌍 찾기
    similar_pairs = find_similar_skills(similarity_df, threshold=0.2)
    
    print(f"\n🔍 유사한 수학 개념 쌍 (유사도 > 0.2):")
    print("=" * 50)
    for _, row in similar_pairs.head(10).iterrows():
        print(f"  {row['skill1']} ↔ {row['skill2']}: {row['similarity']:.3f}")
    
    # 4. 스킬별 통계 정보
    print(f"\n📈 스킬별 통계 정보:")
    print("=" * 50)
    skill_stats = df_clean.groupby('skill_name').agg({
        'correct': ['mean', 'count'],
        'attempt_count': 'mean',
        'ms_first_response': 'mean'
    }).round(3)
    
    skill_stats.columns = ['accuracy', 'problem_count', 'avg_attempts', 'avg_response_time']
    skill_stats = skill_stats.sort_values('problem_count', ascending=False)
    
    print("상위 10개 스킬:")
    print(skill_stats.head(10))
    
    # 5. 유사도 정보를 데이터셋에 추가
    print("\n🔗 유사도 정보를 데이터셋에 추가 중...")
    
    # 5.1 가장 유사한 스킬과 유사도 점수 추가
    def get_most_similar_skill(skill_name, similarity_df, similar_pairs):
        """가장 유사한 스킬과 유사도 점수를 반환"""
        if skill_name not in similarity_df.index:
            return None, 0.0
        
        # 해당 스킬과 가장 유사한 스킬 찾기
        skill_similarities = similarity_df.loc[skill_name].drop(skill_name)
        if len(skill_similarities) == 0:
            return None, 0.0
        
        most_similar_skill = skill_similarities.idxmax()
        max_similarity = skill_similarities.max()
        
        return most_similar_skill, max_similarity
    
    # 5.2 스킬 그룹 생성 (유사도 기반 클러스터링)
    def create_skill_groups(similarity_df, threshold=0.3):
        """유사도 기반으로 스킬 그룹을 생성"""
        from sklearn.cluster import AgglomerativeClustering
        
        # 유사도 매트릭스를 거리 매트릭스로 변환
        distance_matrix = 1 - similarity_df.values
        
        # 클러스터링 수행
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=1-threshold,  # 유사도 0.3 이상이면 같은 그룹
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 스킬명과 클러스터 라벨 매핑
        skill_groups = {}
        for i, skill in enumerate(similarity_df.index):
            skill_groups[skill] = f"Group_{cluster_labels[i]}"
        
        return skill_groups
    
    # 5.3 유사도 기반 난이도 조정
    def calculate_adjusted_difficulty(skill_name, similarity_df, skill_stats):
        """유사 스킬들의 평균 난이도를 계산"""
        if skill_name not in similarity_df.index:
            return skill_stats.get(skill_name, 0.5)
        
        # 유사도 0.3 이상인 스킬들 찾기
        similar_skills = similarity_df.loc[skill_name][similarity_df.loc[skill_name] >= 0.3]
        similar_skills = similar_skills.drop(skill_name)  # 자기 자신 제외
        
        if len(similar_skills) == 0:
            return skill_stats.get(skill_name, 0.5)
        
        # 유사 스킬들의 평균 정답률 계산 (정답률이 높을수록 쉬움)
        similar_accuracies = []
        for similar_skill in similar_skills.index:
            if similar_skill in skill_stats.index:
                similar_accuracies.append(skill_stats.loc[similar_skill, 'accuracy'])
        
        if len(similar_accuracies) == 0:
            return skill_stats.get(skill_name, 0.5)
        
        # 가중평균 (유사도가 높을수록 더 큰 가중치)
        weights = similar_skills.values
        weighted_avg = np.average(similar_accuracies, weights=weights)
        
        return weighted_avg
    
    # 유사도 정보 추가
    print("  - 가장 유사한 스킬과 유사도 점수 계산...")
    df_clean['most_similar_skill'] = None
    df_clean['similarity_score'] = 0.0
    
    for skill in df_clean['skill_name'].unique():
        mask = df_clean['skill_name'] == skill
        most_similar, score = get_most_similar_skill(skill, similarity_df, similar_pairs)
        df_clean.loc[mask, 'most_similar_skill'] = most_similar
        df_clean.loc[mask, 'similarity_score'] = score
    
    # 스킬 그룹 생성
    print("  - 스킬 그룹 생성...")
    skill_groups = create_skill_groups(similarity_df, threshold=0.3)
    df_clean['skill_group'] = df_clean['skill_name'].map(skill_groups)
    
    # 조정된 난이도 계산
    print("  - 유사도 기반 난이도 조정...")
    df_clean['adjusted_difficulty'] = df_clean['skill_name'].apply(
        lambda x: calculate_adjusted_difficulty(x, similarity_df, skill_stats)
    )
    
    # 6. 최종 데이터 정리
    print("\n📋 최종 데이터 정리 중...")
    
    # 필요한 컬럼만 선택 (유사도 정보 포함 + 더 많은 원본 정보)
    final_columns = [
        # 기본 정보
        'order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id', 
        'original', 'correct', 'attempt_count', 'ms_first_response', 
        'tutor_mode', 'answer_type', 'sequence_id', 'student_class_id', 
        'position', 'type', 'base_sequence_id', 'skill_id', 'skill_name', 
        'teacher_id', 'school_id', 'hint_count', 'hint_total', 'overlap_time', 
        'template_id', 'answer_id', 'answer_text', 'first_action', 
        'bottom_hint', 'opportunity', 'opportunity_original',
        # 추가된 유사도 정보
        'most_similar_skill', 'similarity_score', 'skill_group', 'adjusted_difficulty'
    ]
    
    df_final = df_clean[final_columns].copy()
    
    # 7. 통계 정보 출력
    print("\n📊 전처리 결과 통계:")
    print(f"최종 데이터 크기: {df_final.shape}")
    print(f"학생 수: {df_final['user_id'].nunique():,}")
    print(f"문제 수: {df_final['problem_id'].nunique():,}")
    print(f"스킬 수: {df_final['skill_name'].nunique():,}")
    
    print(f"\n🔗 추가된 유사도 정보:")
    print(f"스킬 그룹 수: {df_final['skill_group'].nunique()}")
    print(f"평균 유사도 점수: {df_final['similarity_score'].mean():.3f}")
    print(f"평균 조정된 난이도: {df_final['adjusted_difficulty'].mean():.3f}")
    
    print(f"\n📈 스킬 그룹 분포:")
    print(df_final['skill_group'].value_counts().head(10))
    
    # 8. 파일 저장
    print(f"\n💾 전처리된 데이터 저장 중: {output_file}")
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    # 유사도 매트릭스도 저장
    similarity_file = output_file.replace('.csv', '_similarity_matrix.csv')
    similarity_df.to_csv(similarity_file, encoding='utf-8')
    print(f"유사도 매트릭스 저장: {similarity_file}")
    
    # 유사한 스킬 쌍도 저장
    pairs_file = output_file.replace('.csv', '_similar_pairs.csv')
    similar_pairs.to_csv(pairs_file, index=False, encoding='utf-8')
    print(f"유사한 스킬 쌍 저장: {pairs_file}")
    
    print("✅ 전처리 완료!")
    print("=" * 60)
    
    return df_final, similarity_df, similar_pairs

if __name__ == "__main__":
    # 전처리 실행
    processed_data, similarity_matrix, similar_pairs = preprocess_assistment_model1()
    
    print("\n🎯 전처리된 데이터 샘플:")
    print(processed_data.head())
    
    print("\n🔗 추가된 유사도 정보 샘플:")
    print("=" * 50)
    sample_data = processed_data[['skill_name', 'most_similar_skill', 'similarity_score', 'skill_group', 'adjusted_difficulty']].head(10)
    print(sample_data)
    
    print("\n🔍 수학 개념 유사도 분석 결과:")
    print("=" * 50)
    print(f"총 {len(similar_pairs)}개의 유사한 스킬 쌍 발견!")
    print("상위 15개 유사한 스킬 쌍:")
    print(similar_pairs.head(15))
    
    print(f"\n📊 유사도 매트릭스 크기: {similarity_matrix.shape}")
    print("유사도 매트릭스 샘플 (상위 10x10):")
    print(similarity_matrix.iloc[:10, :10].round(3))
    
    print(f"\n🎯 가장 유사한 수학 개념 쌍 TOP 10:")
    print("=" * 60)
    for i, (_, row) in enumerate(similar_pairs.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['skill1']} ↔ {row['skill2']}: {row['similarity']:.3f}")
