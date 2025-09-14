"""
DKT (Deep Knowledge Tracing) Model Variants:

- dkt-old.py: 기본 DKT 모델 - 문제와 응답만 사용
- dkt2.py (현재 모델): 
    - 난이도 feature 추가 모델
    - 정적 난이도 계산 (전체 데이터 기반)
    - 시도 횟수, 풀이 시간 정규화
    - 문제 풀이 수 기반 가중치 적용
    - 향상된 시각화 기능
- dkt3.py: 다중 feature 통합 및 Z-score 정규화 모델
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
            self.dataset_dir, "skill_builder_data.csv"
        )

        self.q_seqs, self.r_seqs, self.d_seqs, self.s_seqs, self.a_seqs, self.t_seqs, self.q_list, self.u_list, self.q2idx, self.u2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs, self.d_seqs, self.s_seqs, self.a_seqs, self.t_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.d_seqs, self.s_seqs, self.a_seqs, self.t_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return (
            self.q_seqs[index],
            self.r_seqs[index],
            self.d_seqs[index],
            self.s_seqs[index],
            self.a_seqs[index],
            self.t_seqs[index]
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
        
        # 전체 난이도 계산 (정적 난이도) - 개선된 방식
        problem_stats = df.groupby('skill_name').agg({
            'correct': ['count', 'mean'],  # 문제 풀이 수와 평균 정답률
            'attempt_count': 'mean',       # 평균 시도 횟수
            'ms_first_response': 'mean'    # 평균 풀이 시간
        }).reset_index()
        
        # 컬럼명 정리
        problem_stats.columns = ['skill_name', 'solve_count', 'correct_rate', 'attempt_mean', 'time_mean']
        
        # 문제 풀이 수로 가중치 계산 (많이 풀린 문제일수록 신뢰도 높음)
        total_solves = problem_stats['solve_count'].sum()
        problem_stats['weight'] = problem_stats['solve_count'] / total_solves
        
        # 시도 횟수와 풀이 시간을 0-1 사이로 정규화
        problem_stats['attempt_norm'] = (problem_stats['attempt_mean'] - problem_stats['attempt_mean'].min()) / \
                                      (problem_stats['attempt_mean'].max() - problem_stats['attempt_mean'].min())
        problem_stats['time_norm'] = (problem_stats['time_mean'] - problem_stats['time_mean'].min()) / \
                                   (problem_stats['time_mean'].max() - problem_stats['time_mean'].min())
        
        # 난이도 계산: (1-정답률) * 0.6 + 정규화된시도횟수 * 0.2 + 정규화된풀이시간 * 0.2
        problem_stats['static_difficulty'] = (1 - problem_stats['correct_rate']) * 0.6 + \
                                           problem_stats['attempt_norm'] * 0.2 + \
                                           problem_stats['time_norm'] * 0.2
        
        # 가중치를 적용하여 난이도 보정 (많이 풀린 문제의 난이도를 더 신뢰)
        problem_stats['static_difficulty'] = problem_stats['static_difficulty'] * \
                                           (0.8 + 0.2 * problem_stats['weight'])
        
        print("Computing difficulty statistics...")
        
        # 정적 난이도 Z-score 정규화
        problem_stats['static_difficulty_zscore'] = (
            (problem_stats['static_difficulty'] - problem_stats['static_difficulty'].mean()) 
            / problem_stats['static_difficulty'].std()
        )
        
        # 시도 횟수와 풀이 시간도 정규화
        problem_stats['attempt_zscore'] = (
            (problem_stats['attempt_mean'] - problem_stats['attempt_mean'].mean()) 
            / problem_stats['attempt_mean'].std()
        )
        problem_stats['time_zscore'] = (
            (problem_stats['time_mean'] - problem_stats['time_mean'].mean()) 
            / problem_stats['time_mean'].std()
        )

        print("Static difficulty stats:", problem_stats['static_difficulty'].describe())

        # 원본 데이터프레임과 통계 merge
        # 정적 난이도 관련 통계 merge
        df = df.merge(problem_stats, on='skill_name', suffixes=('', '_mean'))
        
        # 이미 계산된 동적 난이도는 df에 있음
        print("\nDifficulty Statistics:")
        print(problem_stats[['skill_name', 'static_difficulty', 'static_difficulty_zscore']].describe())

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []
        d_seqs = []  # 동적 난이도 시퀀스
        s_seqs = []  # 정적 난이도 시퀀스
        a_seqs = []  # 시도 횟수 시퀀스
        t_seqs = []  # 풀이 시간 시퀀스

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_name"]])
            r_seq = df_u["correct"].values
            d_seq = df_u["static_difficulty_zscore"].values  # 정적 난이도 사용
            s_seq = df_u["static_difficulty_zscore"].values  # 정적 난이도 복사
            a_seq = df_u["attempt_zscore"].values
            t_seq = df_u["time_zscore"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            d_seqs.append(d_seq)  # 동적 난이도
            s_seqs.append(s_seq)  # 정적 난이도
            a_seqs.append(a_seq)
            t_seqs.append(t_seq)

        return q_seqs, r_seqs, d_seqs, s_seqs, a_seqs, t_seqs, q_list, u_list, q2idx, u2idx  # s_seqs 추가됨
    
if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


def match_seq_len(q_seqs, r_seqs, d_seqs, s_seqs, a_seqs, t_seqs, seq_len, pad_val=-1):
    proc_q_seqs = []
    proc_r_seqs = []
    proc_d_seqs = []
    proc_s_seqs = []
    proc_a_seqs = []
    proc_t_seqs = []

    for q_seq, r_seq, d_seq, s_seq, a_seq, t_seq in zip(q_seqs, r_seqs, d_seqs, s_seqs, a_seqs, t_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_d_seqs.append(d_seq[i:i + seq_len + 1])
            proc_s_seqs.append(s_seq[i:i + seq_len + 1])
            proc_a_seqs.append(a_seq[i:i + seq_len + 1])
            proc_t_seqs.append(t_seq[i:i + seq_len + 1])

            i += seq_len + 1

        # Padding for remaining sequence
        padding = np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
        proc_q_seqs.append(np.concatenate([q_seq[i:], padding]))
        proc_r_seqs.append(np.concatenate([r_seq[i:], padding]))
        proc_d_seqs.append(np.concatenate([d_seq[i:], padding]))
        proc_s_seqs.append(np.concatenate([s_seq[i:], padding]))
        proc_a_seqs.append(np.concatenate([a_seq[i:], padding]))
        proc_t_seqs.append(np.concatenate([t_seq[i:], padding]))

    return proc_q_seqs, proc_r_seqs, proc_d_seqs, proc_s_seqs, proc_a_seqs, proc_t_seqs

def collate_fn(batch, pad_val=-1):
    ### [그림 1] SAKT 의 Input 과 Output ###
    q_seqs = []
    r_seqs = []
    d_seqs = []  # 동적 난이도
    s_seqs = []  # 정적 난이도
    a_seqs = []  # 시도 횟수
    t_seqs = []  # 풀이 시간
    qshft_seqs = []
    rshft_seqs = []
    dshft_seqs = []
    sshft_seqs = []
    ashft_seqs = []
    tshft_seqs = []

    for data in batch:
        q_seq, r_seq, d_seq, s_seq, a_seq, t_seq = data
        # 현재 시퀀스
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        d_seqs.append(FloatTensor(d_seq[:-1]))
        s_seqs.append(FloatTensor(s_seq[:-1]))
        a_seqs.append(FloatTensor(a_seq[:-1]))
        t_seqs.append(FloatTensor(t_seq[:-1]))
        # 다음 시퀀스
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        dshft_seqs.append(FloatTensor(d_seq[1:]))
        sshft_seqs.append(FloatTensor(s_seq[1:]))
        ashft_seqs.append(FloatTensor(a_seq[1:]))
        tshft_seqs.append(FloatTensor(t_seq[1:]))

    ### E(Exercise)
    q_seqs = pad_sequence(q_seqs, batch_first=True, padding_value=pad_val)
    ### R(Response)
    r_seqs = pad_sequence(r_seqs, batch_first=True, padding_value=pad_val)
    ### D(Difficulty)
    d_seqs = pad_sequence(d_seqs, batch_first=True, padding_value=pad_val)
    ### S(Static Difficulty)
    s_seqs = pad_sequence(s_seqs, batch_first=True, padding_value=pad_val)
    ### A(Attempts)
    a_seqs = pad_sequence(a_seqs, batch_first=True, padding_value=pad_val)
    ### T(Time)
    t_seqs = pad_sequence(t_seqs, batch_first=True, padding_value=pad_val)
    
    ### Shifted sequences
    qshft_seqs = pad_sequence(qshft_seqs, batch_first=True, padding_value=pad_val)
    rshft_seqs = pad_sequence(rshft_seqs, batch_first=True, padding_value=pad_val)
    dshft_seqs = pad_sequence(dshft_seqs, batch_first=True, padding_value=pad_val)
    sshft_seqs = pad_sequence(sshft_seqs, batch_first=True, padding_value=pad_val)
    ashft_seqs = pad_sequence(ashft_seqs, batch_first=True, padding_value=pad_val)
    tshft_seqs = pad_sequence(tshft_seqs, batch_first=True, padding_value=pad_val)

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs = q_seqs * mask_seqs
    r_seqs = r_seqs * mask_seqs
    d_seqs = d_seqs * mask_seqs
    s_seqs = s_seqs * mask_seqs
    a_seqs = a_seqs * mask_seqs
    t_seqs = t_seqs * mask_seqs
    qshft_seqs = qshft_seqs * mask_seqs
    rshft_seqs = rshft_seqs * mask_seqs
    dshft_seqs = dshft_seqs * mask_seqs
    sshft_seqs = sshft_seqs * mask_seqs
    ashft_seqs = ashft_seqs * mask_seqs
    tshft_seqs = tshft_seqs * mask_seqs

    return q_seqs, r_seqs, d_seqs, s_seqs, a_seqs, t_seqs, qshft_seqs, rshft_seqs, dshft_seqs, sshft_seqs, ashft_seqs, tshft_seqs, mask_seqs

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

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
from visualization import visualize_attention_patterns, plot_learning_progress, analyze_skill_mastery
from visualization_global import visualize_global_attention_patterns
from visualization_global_graph import visualize_global_relationship_graph


class SAKT(Module):
    def __init__(self, num_q, n, d, num_attn_heads, dropout):
        super().__init__()
        self.num_q = num_q # 개념(SKILL)의 개수
        self.n = n # 시퀀스 길이
        self.d = d # 임베딩 차원
        self.num_attn_heads = num_attn_heads # head 개수
        self.dropout = dropout # 드롭아웃 비율

        self.M = Embedding(self.num_q * 2, self.d) # Interaction embedding layer
        self.E = Embedding(self.num_q, d) # Exercise embedding layer
        self.P = Parameter(torch.Tensor(self.n, self.d)) # Positional Encoding
        
        # 난이도 관련 임베딩 레이어들
        self.feature_projection = Sequential(
            Linear(3, self.d),  # 3개의 feature (난이도, 시도 횟수, 풀이 시간)
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

    def forward(self, q, r, qry, d_diff, s_diff, att, time):
        ### 1. Model Input ###
        ### [그림 2] Represented Interaction ###
        x = q + self.num_q * r

        ### 2. Embedding ###
        ### 1) Interaction embedding ###
        M = self.M(x).permute(1, 0, 2)
        ### 2) Exercise embedding ###
        E = self.E(qry).permute(1, 0, 2)
        ### 3) Positional encoding ###
        P = self.P.unsqueeze(1)
        
        # 정적 난이도만 사용
        # 난이도 관련 feature 결합
        features = torch.stack([s_diff, att, time], dim=-1)  # [batch_size, seq_len, 3]
        feature_emb = self.feature_projection(features).permute(1, 0, 2)  # [seq_len, batch_size, d]
        
        # Exercise 임베딩과 feature 임베딩 결합
        E = E + feature_emb

        ### [그림 16] Masking Furture Interactions
        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        ### [그림 10] Embedded Interaction Input Matrix
        M = M + P

        ### 3. Self-attention
        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        ### 4. Prediction ###
        ### 1) Feed Forward Network ###
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        ### 2) Prediction Layer ###
        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights

    def train_model(
        self, train_loader, test_loader, num_epochs, opt
    ):

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, d, s, a, t, qshft, rshft, dshft, sshft, ashft, tshft, m = data

                self.train()

                p, _ = self(q.long(), r.long(), qshft.long(), d, s, a, t)
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, d, s, a, t, qshft, rshft, dshft, sshft, ashft, tshft, m = data

                    self.eval()

                    p, _ = self(q.long(), r.long(), qshft.long(), d, s, a, t)
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
                        # 전체 데이터 기반 Attention 패턴 시각화 (히트맵)
                        att_fig = visualize_global_attention_patterns(self, dataset)
                        att_fig.savefig('attention_heatmap_global_dkt2.png', bbox_inches='tight', dpi=300)
                        
                        # 전체 데이터 기반 관계 그래프
                        graph_fig = visualize_global_relationship_graph(self, dataset)
                        graph_fig.savefig('attention_graph_global_dkt2.png', bbox_inches='tight', dpi=300)
                        
                        # 전체 attention 분석만 실행

batch_size = 256
num_epochs = 30  # 원래 30회로 복구
learning_rate = 0.001

model = SAKT(dataset.num_q, n=100, d=100, num_attn_heads=5, dropout=0.2).to("cpu")

opt = Adam(model.parameters(), learning_rate)

model.train_model(
    train_loader, test_loader, num_epochs, opt
)