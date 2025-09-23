"""
DKT (Deep Knowledge Tracing) Model with Self-Attention for Duolingo SLAM Dataset
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

class Preprocessor(Dataset):
    def __init__(self, seq_len) -> None:
        super().__init__()
        self.dataset_path = "processed_dataverse_data.csv"
        self.q_seqs, self.r_seqs, self.d_seqs, self.q_list, self.u_list, self.q2idx, self.u2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]  # 사용자 수
        self.num_q = self.q_list.shape[0]  # 문제 수

        if seq_len:
            self.q_seqs, self.r_seqs, self.d_seqs = match_seq_len(self.q_seqs, self.r_seqs, self.d_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return (
            self.q_seqs[index],
            self.r_seqs[index],
            self.d_seqs[index]
        )

    def __len__(self):
        return self.len

    def preprocess(self):
        print("Loading data from:", self.dataset_path)
        df = pd.read_csv(self.dataset_path)
        print("Initial data shape:", df.shape)
        
        # 정렬 (시간순)
        df = df.sort_values(by=["user_id", "time"])
        print("Data columns:", df.columns.tolist())

        # 사용자 수 제한 해제 (전체 데이터 사용)
        unique_users = df["user_id"].unique()
        print(f"Using all {len(unique_users)} users for full training.")

        # 문제별 통계 계산 (정답률, 누적시도, 풀이시간, 난이도)
        problem_stats = df.groupby('item_id').agg({
            'correct': ['count', 'mean'],
            'cumulative_attempts': 'mean',
            'time': 'mean',
            'difficulty': 'mean'
        }).reset_index()
        problem_stats.columns = ['item_id', 'solve_count', 'correct_rate', 'attempt_mean', 'time_mean', 'difficulty_mean']
        total_solves = problem_stats['solve_count'].sum()
        problem_stats['weight'] = problem_stats['solve_count'] / total_solves
        # 정규화
        problem_stats['attempt_norm'] = (problem_stats['attempt_mean'] - problem_stats['attempt_mean'].min()) / \
                                      (problem_stats['attempt_mean'].max() - problem_stats['attempt_mean'].min() + 1e-8)
        problem_stats['time_norm'] = (problem_stats['time_mean'] - problem_stats['time_mean'].min()) / \
                                   (problem_stats['time_mean'].max() - problem_stats['time_mean'].min() + 1e-8)
        # 난이도 계산 (difficulty 컬럼 제외, 정답률 가중치 0.6)
        problem_stats['static_difficulty'] = (1 - problem_stats['correct_rate']) * 0.6 + \
                        problem_stats['attempt_norm'] * 0.2 + \
                        problem_stats['time_norm'] * 0.2
        problem_stats['static_difficulty'] = problem_stats['static_difficulty'] * (0.8 + 0.2 * problem_stats['weight'])
        # Z-score 정규화
        problem_stats['static_difficulty_zscore'] = (
            (problem_stats['static_difficulty'] - problem_stats['static_difficulty'].mean()) 
            / (problem_stats['static_difficulty'].std() + 1e-8)
        )
        print("Static difficulty stats:", problem_stats['static_difficulty'].describe())
        # merge
        df = df.merge(problem_stats[['item_id', 'static_difficulty_zscore']], on='item_id', how='left')

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["item_id"].values)
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        q_seqs = []
        r_seqs = []
        d_seqs = []
        for u in u_list:
            df_u = df[df["user_id"] == u]
            q_seq = np.array([q2idx[q] for q in df_u["item_id"]])
            r_seq = df_u["correct"].values
            d_seq = df_u["static_difficulty_zscore"].values.astype(np.float32)
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            d_seqs.append(d_seq)
        return q_seqs, r_seqs, d_seqs, q_list, u_list, q2idx, u2idx

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

def match_seq_len(q_seqs, r_seqs, d_seqs, seq_len, pad_val=-1):
    proc_q_seqs = []
    proc_r_seqs = []
    proc_d_seqs = []

    for q_seq, r_seq, d_seq in zip(q_seqs, r_seqs, d_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_d_seqs.append(d_seq[i:i + seq_len + 1])
            i += seq_len + 1

        # Padding for remaining sequence
        padding = np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
        proc_q_seqs.append(np.concatenate([q_seq[i:], padding]))
        proc_r_seqs.append(np.concatenate([r_seq[i:], padding]))
        proc_d_seqs.append(np.concatenate([d_seq[i:], padding]))

    return proc_q_seqs, proc_r_seqs, proc_d_seqs

def collate_fn(batch, pad_val=-1):
    q_seqs = []
    r_seqs = []
    d_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    dshft_seqs = []

    for data in batch:
        q_seq, r_seq, d_seq = data
        # 현재 시퀀스
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        d_seqs.append(FloatTensor(d_seq[:-1]))
        # 다음 시퀀스
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        dshft_seqs.append(FloatTensor(d_seq[1:]))

    q_seqs = pad_sequence(q_seqs, batch_first=True, padding_value=pad_val)
    r_seqs = pad_sequence(r_seqs, batch_first=True, padding_value=pad_val)
    d_seqs = pad_sequence(d_seqs, batch_first=True, padding_value=pad_val)
    qshft_seqs = pad_sequence(qshft_seqs, batch_first=True, padding_value=pad_val)
    rshft_seqs = pad_sequence(rshft_seqs, batch_first=True, padding_value=pad_val)
    dshft_seqs = pad_sequence(dshft_seqs, batch_first=True, padding_value=pad_val)

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs = q_seqs * mask_seqs
    r_seqs = r_seqs * mask_seqs
    d_seqs = d_seqs * mask_seqs
    qshft_seqs = qshft_seqs * mask_seqs
    rshft_seqs = rshft_seqs * mask_seqs
    dshft_seqs = dshft_seqs * mask_seqs

    return q_seqs, r_seqs, d_seqs, qshft_seqs, rshft_seqs, dshft_seqs, mask_seqs

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
from visualization import visualize_attention_patterns, plot_learning_progress, analyze_skill_mastery
from visualization_slam import visualize_slam_attention_patterns

# MPS 호환 MultiheadAttention 구현
class MPSCompatibleMultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.dropout = Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, embed_dim = query.size()
        
        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final linear projection
        attn_output = self.out_proj(attn_output)
        
        # Average attention weights across heads for compatibility
        avg_attn_weights = attn_weights.mean(dim=1)
        
        return attn_output, avg_attn_weights
from visualization_global_graph import visualize_global_relationship_graph
from visualization_slam_enhanced import visualize_slam_attention_patterns_enhanced

class SAKT(Module):
    def __init__(self, num_q, n, d, num_attn_heads, dropout):
        super().__init__()
        self.num_q = num_q # 문제의 개수
        self.n = n # 시퀀스 길이
        self.d = d # 임베딩 차원
        self.num_attn_heads = num_attn_heads # head 개수
        self.dropout = dropout # 드롭아웃 비율

        self.M = Embedding(self.num_q * 2, self.d) # Interaction embedding layer
        self.E = Embedding(self.num_q, d) # Exercise embedding layer
        self.P = Parameter(torch.Tensor(self.n, self.d)) # Positional Encoding
        
        # 난이도 feature 투영 레이어 추가
        self.difficulty_projection = Sequential(
            Linear(1, self.d),  # 1개의 feature (난이도만)
            ReLU(),
            Dropout(self.dropout)
        )

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

    def forward(self, q, r, qry, d_diff):
        ### 1. Model Input ###
        x = q + self.num_q * r

        ### 2. Embedding ###
        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)
        
        # 난이도 feature 임베딩 추가
        difficulty_emb = self.difficulty_projection(d_diff.unsqueeze(-1)).permute(1, 0, 2)
        E = E + difficulty_emb  # Exercise 임베딩에 난이도 정보 추가

        ### Masking Future Interactions (MPS 호환)
        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]], device=E.device), diagonal=1
        ).bool()

        ### Embedded Interaction Input Matrix
        M = M + P

        ### 3. Self-attention
        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
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



    def train_model(self, train_loader, test_loader, num_epochs, opt):
        from tqdm import tqdm
        max_auc = 0
        dataset = train_loader.dataset.dataset  # Get the original dataset
        device = next(self.parameters()).device  # 모델의 디바이스 가져오기

        for i in range(1, num_epochs + 1):
            loss_mean = []

            print(f"Epoch {i} training...")
            for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {i}")):
                q, r, d, qshft, rshft, dshft, m = data
                
                # 데이터를 GPU로 이동
                q, r, d = q.to(device), r.to(device), d.to(device)
                qshft, rshft, dshft = qshft.to(device), rshft.to(device), dshft.to(device)
                m = m.to(device)

                self.train()

                p, _ = self(q.long(), r.long(), qshft.long(), dshft)
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

                # 미니배치별 로그 출력
                if batch_idx % 10 == 0:
                    print(f"  [Epoch {i} | Batch {batch_idx}] Loss: {loss.item():.4f}")

            with torch.no_grad():
                for data in test_loader:
                    q, r, d, qshft, rshft, dshft, m = data
                    
                    # 검증 데이터도 GPU로 이동
                    q, r, d = q.to(device), r.to(device), d.to(device)
                    qshft, rshft, dshft = qshft.to(device), rshft.to(device), dshft.to(device)
                    m = m.to(device)

                    self.eval()

                    p, _ = self(q.long(), r.long(), qshft.long(), dshft)
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
                att_fig, basic_attention_matrix, original_attention_matrix, problem_labels = visualize_slam_attention_patterns(self, dataset, return_matrix=True)
                att_fig.savefig('attention_heatmap_global_self_attention_slam_basic.png', bbox_inches='tight', dpi=300)
                print("Basic global attention visualization saved")
                
                # 개선된 전체 데이터 기반 Attention 패턴 시각화 (Basic 매트릭스 사용)
                try:
                    enhanced_fig, original_matrix, normalized_matrix = visualize_slam_attention_patterns_enhanced(self, dataset, basic_matrix=basic_attention_matrix, original_matrix=original_attention_matrix, problem_names=problem_labels)
                    enhanced_fig.savefig('attention_heatmap_global_self_attention_slam_enhanced.png', bbox_inches='tight', dpi=300)
                    print("Enhanced global attention visualization saved")
                except Exception as e:
                    print(f"Enhanced visualization error: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 전체 데이터 기반 관계 그래프
                graph_fig = visualize_global_relationship_graph(self, dataset)
                graph_fig.savefig('attention_graph_global_self_attention_slam.png', bbox_inches='tight', dpi=300)
                print("Global relationship graph saved")
                
                # 개별 어텐션 시각화는 파라미터 불일치로 인해 제거
                print("Individual attention visualization skipped (parameter mismatch)")
                
                # 개선된 결과 요약
                print(f"\n=== 원본값 vs 정규화 비교 분석 결과 ===")
                print(f"원본 어텐션 범위: {original_matrix.min():.6f} ~ {original_matrix.max():.6f}")
                print(f"평균 어텐션: {original_matrix.mean():.6f}")
                print(f"표준편차: {original_matrix.std():.6f}")
                print(f"비교 방식: 동일한 어텐션 값의 원본 vs 정규화 표시")
                
            except Exception as e:
                print(f"Visualization error: {e}")
                print("Skipping visualization due to error")

# 데이터 로드 및 모델 학습
print("Loading dataset...")
dataset = Preprocessor(seq_len=100)

print(f"Total number of sequences: {len(dataset)}")
print(f"Number of unique problems: {dataset.num_q}")
print(f"Number of unique students: {dataset.num_u}")

# Train/Test 분할 (80/20)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size]
)

# 데이터 로더 설정
train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=test_size, shuffle=True,
    collate_fn=collate_fn
)

# 하이퍼파라미터 설정
batch_size = 512  # GPU 메모리에 맞게 증가
num_epochs = 1  # 테스트용 1 에포크
learning_rate = 0.001

# CPU 전용 모드 (안정성 우선)
device = torch.device("cpu")
print("⚡ CPU 전용 모드로 실행됨 (안정성 우선)")
print(f"사용 디바이스: {device}")

# 모델 초기화 및 학습
print("Initializing model...")
model = SAKT(dataset.num_q, n=100, d=100, num_attn_heads=5, dropout=0.2).to(device)
opt = Adam(model.parameters(), learning_rate)

print("Starting training...")
model.train_model(
    train_loader, test_loader, num_epochs, opt
)
