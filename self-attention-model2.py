"""
Self-Attentive Knowledge Tracing (SAKT) Model Implementation - Model 2
Based on the paper: "Self-Attentive Knowledge Tracing" by Ghosh et al.
Enhanced with advanced mathematical concept similarity features

Model 2 Features:
- 기본 DKT 모델 구현
- 문제(skill)와 응답(correct/incorrect)만을 입력으로 사용
- Self-Attention 메커니즘 적용
- 고급 수학 개념 유사도 정보 활용
- 스킬 그룹 기반 학습 패턴 분석
- 기본적인 시각화 기능 (attention 패턴, 관계 그래프)
"""

import os
import argparse
import json
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset

DATASET_DIR = ""

class Preprocessor(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, "skill_builder_model1_processed.csv"
        )

        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, self.u2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs = match_seq_len(self.q_seqs, self.r_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return (
            self.q_seqs[index],
            self.r_seqs[index]
        )

    def __len__(self):
        return self.len

    def preprocess(self):
        print("Loading data from:", self.dataset_path)
        df = pd.read_csv(self.dataset_path, encoding='unicode_escape', low_memory=False)
        print("Initial data shape:", df.shape)
        
        df = df.dropna(subset=["skill_name"])
        print("After dropping NA:", df.shape)
        
        df = df.drop_duplicates(subset=["order_id", "skill_name"])
        print("After dropping duplicates:", df.shape)
        
        df = df.sort_values(by=["order_id"])
        print("Data columns:", df.columns.tolist())

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_name"]])
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

def match_seq_len(q_seqs, r_seqs, seq_len, pad_val=-1):
    proc_q_seqs = []
    proc_r_seqs = []

    for q_seq, r_seq in zip(q_seqs, r_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            i += seq_len + 1

        # Padding for remaining sequence
        padding = np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
        proc_q_seqs.append(np.concatenate([q_seq[i:], padding]))
        proc_r_seqs.append(np.concatenate([r_seq[i:], padding]))

    return proc_q_seqs, proc_r_seqs

def collate_fn(batch, pad_val=-1):
    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []

    for data in batch:
        q_seq, r_seq = data
        # 현재 시퀀스
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        # 다음 시퀀스
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))

    q_seqs = pad_sequence(q_seqs, batch_first=True, padding_value=pad_val)
    r_seqs = pad_sequence(r_seqs, batch_first=True, padding_value=pad_val)
    qshft_seqs = pad_sequence(qshft_seqs, batch_first=True, padding_value=pad_val)
    rshft_seqs = pad_sequence(rshft_seqs, batch_first=True, padding_value=pad_val)

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs = q_seqs * mask_seqs
    r_seqs = r_seqs * mask_seqs
    qshft_seqs = qshft_seqs * mask_seqs
    rshft_seqs = rshft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
from visualization import visualize_attention_patterns, plot_learning_progress, analyze_skill_mastery
from visualization_skillbuilder import visualize_skillbuilder_attention_patterns
from visualization_skillbuilder_enhanced import visualize_skillbuilder_attention_patterns_enhanced
from visualization_global_graph import visualize_global_relationship_graph

class SAKT(Module):
    def __init__(self, num_q, n, d, num_attn_heads, dropout, similarity_matrix=None, skill_groups=None):
        super().__init__()
        self.num_q = num_q # 개념(SKILL)의 개수
        self.n = n # 시퀀스 길이
        self.d = d # 임베딩 차원
        self.num_attn_heads = num_attn_heads # head 개수
        self.dropout = dropout # 드롭아웃 비율

        self.M = Embedding(self.num_q * 2, self.d) # Interaction embedding layer
        self.E = Embedding(self.num_q, d) # Exercise embedding layer
        self.P = Parameter(torch.Tensor(self.n, self.d)) # Positional Encoding

        kaiming_normal_(self.P)

        # Multi head Attention
        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout=self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        # Feed Forward Network
        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)

        # Prediction Layer
        self.pred = Linear(self.d, 1)
        
        # 유사도 정보 저장 (Model 2 전용)
        self.similarity_matrix = similarity_matrix
        self.skill_groups = skill_groups
        
        # 유사도 기반 가중치 계산을 위한 파라미터
        self.similarity_weight = Parameter(torch.tensor(0.1))  # 유사도 가중치
        self.group_weight = Parameter(torch.tensor(0.1))       # 그룹 가중치

    def forward(self, q, r, qry):
        ### 1. Model Input ###
        x = q + self.num_q * r

        ### 2. Embedding ###
        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        ### Masking Future Interactions
        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        ### Embedded Interaction Input Matrix
        M = M + P

        ### 3. Self-attention
        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        
        ### 3.5. 유사도 기반 가중치 적용 (Model 2 전용) ###
        if self.similarity_matrix is not None and self.skill_groups is not None:
            # 유사도 기반 가중치 계산 (이전 유사 문제 성과 포함)
            similarity_weights = self.calculate_similarity_weights(q, r, qry)
            
            # Attention 가중치에 유사도 정보 반영
            S = S * similarity_weights.unsqueeze(-1)
        
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        ### 4. Prediction ###
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)
        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights
    
    def calculate_similarity_weights(self, q, r, qry):
        """
        유사도 정보와 이전 유사 문제 성과를 활용한 가중치 계산 (Model 2 전용)
        """
        batch_size, seq_len = q.shape
        
        # 유사도 가중치 초기화
        similarity_weights = torch.ones_like(q, dtype=torch.float32)
        
        if self.similarity_matrix is not None:
            # 각 시퀀스에 대해 유사도 가중치 계산
            for b in range(batch_size):
                for t in range(seq_len):
                    current_skill = q[b, t].item()
                    query_skill = qry[b, t].item()
                    current_response = r[b, t].item()
                    
                    # 유효한 스킬 ID인지 확인
                    if (current_skill < len(self.similarity_matrix) and 
                        query_skill < len(self.similarity_matrix) and
                        current_skill >= 0 and query_skill >= 0):
                        
                        # 유사도 점수 가져오기
                        similarity_score = self.similarity_matrix.iloc[current_skill, query_skill]
                        
                        # 이전 유사 문제들의 성과 확인
                        performance_weight = self.calculate_performance_weight(
                            q[b], r[b], current_skill, t, similarity_score
                        )
                        
                        # 유사도 기반 가중치 + 성과 기반 가중치 적용
                        similarity_weights[b, t] = (1.0 + 
                            self.similarity_weight * similarity_score + 
                            performance_weight)
        
        if self.skill_groups is not None:
            # 스킬 그룹 기반 가중치 계산
            group_weights = torch.ones_like(q, dtype=torch.float32)
            
            for b in range(batch_size):
                for t in range(seq_len):
                    current_skill = q[b, t].item()
                    query_skill = qry[b, t].item()
                    
                    # 유효한 스킬 ID인지 확인
                    if (current_skill < len(self.skill_groups) and 
                        query_skill < len(self.skill_groups) and
                        current_skill >= 0 and query_skill >= 0):
                        
                        # 같은 그룹인지 확인
                        current_group = self.skill_groups[current_skill]
                        query_group = self.skill_groups[query_skill]
                        
                        if current_group == query_group:
                            group_weights[b, t] = 1.0 + self.group_weight
            
            # 그룹 가중치와 유사도 가중치 결합
            similarity_weights = similarity_weights * group_weights
        
        return similarity_weights
    
    def calculate_performance_weight(self, q_seq, r_seq, current_skill, current_time, similarity_score):
        """
        이전 유사 문제들의 성과를 기반으로 가중치 계산
        
        Args:
            q_seq: 문제 시퀀스
            r_seq: 응답 시퀀스  
            current_skill: 현재 문제 스킬 ID
            current_time: 현재 시간 인덱스
            similarity_score: 유사도 점수
            
        Returns:
            performance_weight: 성과 기반 가중치
        """
        performance_weight = 0.0
        
        # 현재 시간 이전의 문제들만 확인
        for t in range(current_time):
            past_skill = q_seq[t].item()
            past_response = r_seq[t].item()
            
            # 유효한 응답인지 확인 (0 또는 1)
            if past_response in [0.0, 1.0]:
                # 유사도 매트릭스에서 유사도 점수 가져오기
                if (past_skill < len(self.similarity_matrix) and 
                    current_skill < len(self.similarity_matrix) and
                    past_skill >= 0 and current_skill >= 0):
                    
                    past_similarity = self.similarity_matrix.iloc[past_skill, current_skill]
                    
                    # 유사도가 높은 문제들만 고려 (임계값: 0.3)
                    if past_similarity > 0.3:
                        # 맞춘 경우: 양의 가중치, 틀린 경우: 음의 가중치
                        if past_response == 1.0:  # 맞춤
                            performance_weight += past_similarity * 0.1  # 유사도에 비례한 보너스
                        else:  # 틀림
                            performance_weight -= past_similarity * 0.05  # 유사도에 비례한 페널티
        
        return performance_weight

    def train_model(self, train_loader, test_loader, num_epochs, opt):
        from tqdm import tqdm
        max_auc = 0
        dataset = train_loader.dataset.dataset  # Get the original dataset
        device = next(self.parameters()).device  # 모델의 디바이스 가져오기

        for i in range(1, num_epochs + 1):
            loss_mean = []

            print(f"Epoch {i} training...")
            for data in tqdm(train_loader, desc=f"Epoch {i}"):
                q, r, qshft, rshft, m = data
                
                # 데이터를 GPU로 이동
                q, r = q.to(device), r.to(device)
                qshft, rshft = qshft.to(device), rshft.to(device)
                m = m.to(device)

                self.train()

                p, _ = self(q.long(), r.long(), qshft.long())
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data
                    
                    # 검증 데이터도 GPU로 이동
                    q, r = q.to(device), r.to(device)
                    qshft, rshft = qshft.to(device), rshft.to(device)
                    m = m.to(device)

                    self.eval()

                    p, _ = self(q.long(), r.long(), qshft.long())
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=p.numpy()
                    )

                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {},   Loss Mean: {}"
                        .format(i, auc, loss_mean)
                    )
                    
                    # 마지막 에포크에서 시각화 수행
                    if i == num_epochs:
                        try:
                            # 기존 전체 데이터 기반 Attention 패턴 시각화 (히트맵)
                            att_fig, basic_attention_matrix, original_attention_matrix, problem_labels = visualize_skillbuilder_attention_patterns(self, dataset, return_matrix=True, model_name='model2')
                            att_fig.savefig('attention_heatmap_global_model2_basic.png', bbox_inches='tight', dpi=300)
                            print("Basic global attention visualization saved")
                            
                            # 개선된 전체 데이터 기반 Attention 패턴 시각화
                            enhanced_fig, original_matrix, normalized_matrix = visualize_skillbuilder_attention_patterns_enhanced(self, dataset, basic_matrix=basic_attention_matrix, original_matrix=original_attention_matrix, problem_names=problem_labels)
                            enhanced_fig.savefig('attention_heatmap_global_model2_enhanced.png', bbox_inches='tight', dpi=300)
                            print("Enhanced global attention visualization saved")
                            
                            # 전체 데이터 기반 관계 그래프
                            graph_fig = visualize_global_relationship_graph(self, dataset)
                            graph_fig.savefig('attention_graph_global_model2.png', bbox_inches='tight', dpi=300)
                            print("Global relationship graph saved")
                            
                            # 개선된 결과 요약
                            print(f"\n=== Model 2 개선된 영향도 분석 결과 ===")
                            print(f"원본 영향도 범위: {original_matrix.min():.6f} ~ {original_matrix.max():.6f}")
                            print(f"평균 영향도: {original_matrix.mean():.6f}")
                            print(f"표준편차: {original_matrix.std():.6f}")
                            print(f"개선사항: 시간가중치 + 정답가중치 + 난이도가중치 적용")
                            
                        except Exception as e:
                            print(f"Visualization error: {e}")
                            print("Skipping visualization due to error")
                        
                        # 기존 시각화들도 유지
                        att_fig1, att_fig2 = visualize_attention_patterns(self, dataset)
                        att_fig1.savefig('attention_heatmap_dkt_old.png')
                        att_fig2.savefig('attention_graph_dkt_old.png')
                        
                        # 학습 진행 상황과 스킬 숙련도는 일단 생략
                        # prog_fig = plot_learning_progress(self, dataset)
                        # prog_fig.savefig('learning_progress_dkt_old.png')
                        
                        # skill_fig = analyze_skill_mastery(self, dataset)
                        # skill_fig.savefig('skill_mastery_dkt_old.png')

dataset = Preprocessor(seq_len=100)

print(f"Total number of sequences: {len(dataset)}")
print(f"Number of unique problems: {dataset.num_q}")
print(f"Number of unique students: {dataset.num_u}")

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=test_size, shuffle=True,
    collate_fn=collate_fn
)

batch_size = 512  # MPS용 배치 크기 증가
num_epochs = 30  # 전체 학습으로 복구
learning_rate = 0.001

# CPU 전용 모드 (안정성 우선)
device = torch.device("cpu")
print("⚡ CPU 전용 모드로 실행됨 (안정성 우선)")
print(f"사용 디바이스: {device}")

# 유사도 정보 로드 (Model 2 전용)
similarity_matrix = None
skill_groups = None

try:
    # 유사도 매트릭스 로드
    similarity_df = pd.read_csv('skill_builder_model1_processed_similarity_matrix.csv', index_col=0)
    similarity_matrix = similarity_df
    
    # 스킬 그룹 정보 로드
    processed_df = pd.read_csv('skill_builder_model1_processed.csv')
    skill_groups = processed_df.groupby('skill_name')['skill_group'].first().to_dict()
    
    print("✅ 유사도 정보 로드 완료 (Model 2)")
    print(f"유사도 매트릭스 크기: {similarity_matrix.shape}")
    print(f"스킬 그룹 수: {len(set(skill_groups.values()))}")
except Exception as e:
    print(f"⚠️ 유사도 정보 로드 실패: {e}")
    print("기본 SAKT 모델로 실행됩니다.")

model = SAKT(dataset.num_q, n=100, d=100, num_attn_heads=5, dropout=0.2, 
             similarity_matrix=similarity_matrix, skill_groups=skill_groups).to(device)
opt = Adam(model.parameters(), learning_rate)

model.train_model(
    train_loader, test_loader, num_epochs, opt
)
