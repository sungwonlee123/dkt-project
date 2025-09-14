"""
DKT (Deep Knowledge Tracing) Model Variants:

- dkt-old.py: 기본 DKT 모델 - 기존의 기본적인 Knowledge Tracing 구현
- dkt2.py: 난이도 feature 추가 - 문제의 난이도를 고려한 개선 모델
- dkt3.py: 
    - 난이도, 시도 횟수, 풀이 시간 등 다중 feature 통합
    - Feature들의 Z-score 정규화 적용
    - 향상된 시각화 기능 (전역 attention 패턴, 관계 그래프)
    - 개선된 학습 진행 상황 및 스킬 숙련도 분석
- dkt4.py (현재 모델):
    - AI Hub 수학 문제 데이터셋 적용
    - IRT (Item Response Theory) 파라미터 통합:
        - 문항 난이도(difficulty)
        - 문항 변별도(discrimination)
        - 학생 능력치(user_ability)
        - 학생 일관성(user_consistency)
    - 수학 개념 관계 정보 활용:
        - 선수 개념(prerequisites)
        - 후속 개념(postrequisites)
    - 학습 진행 통계 추가:
        - 누적 시도 수
        - 누적 정답률
    - 개선된 시각화:
        - 개념 관계 네트워크
        - 학습자 진행 경로
        - 개념별 숙달도
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
            self.dataset_dir, "processed_dkt_data.csv"
        )
        self.q_seqs, self.r_seqs, self.d_seqs, self.a_seqs, self.t_seqs, self.q_list, self.u_list, self.q2idx, self.u2idx = self.preprocess()

        self.num_u = len(self.u_list)
        self.num_q = len(self.q_list)

        if seq_len:
            self.q_seqs, self.r_seqs, self.d_seqs, self.a_seqs, self.t_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.d_seqs, self.a_seqs, self.t_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return (
            self.q_seqs[index],
            self.r_seqs[index],
            self.d_seqs[index],
            self.a_seqs[index],
            self.t_seqs[index]
        )

    def __len__(self):
        return self.len

    def preprocess(self):
        print("Loading data from:", self.dataset_path)
        df = pd.read_csv(self.dataset_path)
        print("Initial data shape:", df.shape)

        # 유니크한 유저와 문제 ID 추출
        u_list = df['user_id'].unique()
        q_list = df['item_id'].unique()

        # ID를 인덱스로 매핑
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        # 시퀀스 데이터 생성
        q_seqs = []  # 문제 ID
        r_seqs = []  # 정답 여부
        d_seqs = []  # 난이도
        a_seqs = []  # 누적 시도
        t_seqs = []  # 누적 정답률

        # 사용자별로 시퀀스 생성
        for u in u_list:
            df_u = df[df['user_id'] == u].sort_values('position')
            
            q_seq = np.array([q2idx[q] for q in df_u['item_id']])
            r_seq = df_u['correct'].values
            d_seq = df_u['difficulty'].values
            a_seq = df_u['cumulative_attempts'].values
            t_seq = df_u['running_accuracy'].values

            if len(q_seq) > 0:  # 빈 시퀀스 제외
                q_seqs.append(q_seq)
                r_seqs.append(r_seq)
                d_seqs.append(d_seq)
                a_seqs.append(a_seq)
                t_seqs.append(t_seq)

        return q_seqs, r_seqs, d_seqs, a_seqs, t_seqs, q_list, u_list, q2idx, u2idx

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

def match_seq_len(q_seqs, r_seqs, d_seqs, a_seqs, t_seqs, seq_len, pad_val=-1):
    proc_q_seqs = []
    proc_r_seqs = []
    proc_d_seqs = []
    proc_a_seqs = []
    proc_t_seqs = []

    for q_seq, r_seq, d_seq, a_seq, t_seq in zip(q_seqs, r_seqs, d_seqs, a_seqs, t_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_d_seqs.append(d_seq[i:i + seq_len + 1])
            proc_a_seqs.append(a_seq[i:i + seq_len + 1])
            proc_t_seqs.append(t_seq[i:i + seq_len + 1])

            i += seq_len + 1

        # 남은 시퀀스에 대한 패딩
        if i < len(q_seq):
            padding = np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
            proc_q_seqs.append(np.concatenate([q_seq[i:], padding]))
            proc_r_seqs.append(np.concatenate([r_seq[i:], padding]))
            proc_d_seqs.append(np.concatenate([d_seq[i:], padding]))
            proc_a_seqs.append(np.concatenate([a_seq[i:], padding]))
            proc_t_seqs.append(np.concatenate([t_seq[i:], padding]))

    return proc_q_seqs, proc_r_seqs, proc_d_seqs, proc_a_seqs, proc_t_seqs

def collate_fn(batch, pad_val=-1):
    q_seqs = []
    r_seqs = []
    d_seqs = []
    a_seqs = []
    t_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    dshft_seqs = []
    ashft_seqs = []
    tshft_seqs = []

    for data in batch:
        q_seq, r_seq, d_seq, a_seq, t_seq = data
        
        # 현재 시퀀스
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        d_seqs.append(FloatTensor(d_seq[:-1]))
        a_seqs.append(FloatTensor(a_seq[:-1]))
        t_seqs.append(FloatTensor(t_seq[:-1]))
        
        # 다음 시퀀스 (예측 대상)
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        dshft_seqs.append(FloatTensor(d_seq[1:]))
        ashft_seqs.append(FloatTensor(a_seq[1:]))
        tshft_seqs.append(FloatTensor(t_seq[1:]))

    # 패딩
    q_seqs = pad_sequence(q_seqs, batch_first=True, padding_value=pad_val)
    r_seqs = pad_sequence(r_seqs, batch_first=True, padding_value=pad_val)
    d_seqs = pad_sequence(d_seqs, batch_first=True, padding_value=pad_val)
    a_seqs = pad_sequence(a_seqs, batch_first=True, padding_value=pad_val)
    t_seqs = pad_sequence(t_seqs, batch_first=True, padding_value=pad_val)
    
    qshft_seqs = pad_sequence(qshft_seqs, batch_first=True, padding_value=pad_val)
    rshft_seqs = pad_sequence(rshft_seqs, batch_first=True, padding_value=pad_val)
    dshft_seqs = pad_sequence(dshft_seqs, batch_first=True, padding_value=pad_val)
    ashft_seqs = pad_sequence(ashft_seqs, batch_first=True, padding_value=pad_val)
    tshft_seqs = pad_sequence(tshft_seqs, batch_first=True, padding_value=pad_val)

    # 마스킹 (패딩된 부분 처리)
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs = q_seqs * mask_seqs
    r_seqs = r_seqs * mask_seqs
    d_seqs = d_seqs * mask_seqs
    a_seqs = a_seqs * mask_seqs
    t_seqs = t_seqs * mask_seqs
    qshft_seqs = qshft_seqs * mask_seqs
    rshft_seqs = rshft_seqs * mask_seqs
    dshft_seqs = dshft_seqs * mask_seqs
    ashft_seqs = ashft_seqs * mask_seqs
    tshft_seqs = tshft_seqs * mask_seqs

    return q_seqs, r_seqs, d_seqs, a_seqs, t_seqs, qshft_seqs, rshft_seqs, dshft_seqs, ashft_seqs, tshft_seqs, mask_seqs

# 데이터셋 준비
dataset = Preprocessor(seq_len=100)

print(f"Total number of sequences: {len(dataset)}")
print(f"Number of unique problems: {dataset.num_q}")
print(f"Number of unique students: {dataset.num_u}")

# 학습/테스트 데이터 분할
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
        self.num_q = num_q  # 문제 개수
        self.n = n          # 시퀀스 길이
        self.d = d          # 임베딩 차원
        self.num_attn_heads = num_attn_heads  # attention head 개수
        self.dropout = dropout  # 드롭아웃 비율

        # 문제와 반응에 대한 임베딩
        self.M = Embedding(self.num_q * 2, self.d)  # 문제-반응 상호작용 임베딩
        self.E = Embedding(self.num_q, self.d)      # 문제 임베딩
        self.P = Parameter(torch.Tensor(self.n, self.d))  # 위치 인코딩
        
        # Feature 투영 레이어 (난이도, 시도 횟수, 정답률)
        self.feature_projection = Sequential(
            Linear(3, self.d),
            ReLU(),
            Dropout(self.dropout)
        )

        kaiming_normal_(self.P)

        # Multi-head Attention
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

        # 예측 레이어
        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry, diff, att, time):
        # 문제-반응 상호작용 인덱스
        x = q + self.num_q * r

        # 임베딩
        M = self.M(x).permute(1, 0, 2)  # 상호작용 임베딩
        E = self.E(qry).permute(1, 0, 2)  # 문제 임베딩
        P = self.P.unsqueeze(1)  # 위치 인코딩
        
        # Feature 임베딩
        features = torch.stack([diff, att, time], dim=-1)
        feature_emb = self.feature_projection(features).permute(1, 0, 2)
        
        # 문제 임베딩과 feature 임베딩 결합
        E = E + feature_emb

        # Causal 마스킹 (미래 정보 차단)
        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        # 위치 정보 추가
        M = M + P

        # Self-attention
        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        # Feed Forward Network
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        # 최종 예측
        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights

    def train_model(self, train_loader, test_loader, num_epochs, opt):
        max_auc = 0
        for i in range(1, num_epochs + 1):
            loss_mean = []

            # 학습
            for data in train_loader:
                q, r, d, a, t, qshft, rshft, dshft, ashft, tshft, m = data
                self.train()

                p, _ = self(q.long(), r.long(), qshft.long(), d, a, t)
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            # 평가
            with torch.no_grad():
                for data in test_loader:
                    q, r, d, a, t, qshft, rshft, dshft, ashft, tshft, m = data
                    self.eval()

                    p, _ = self(q.long(), r.long(), qshft.long(), d, a, t)
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=p.numpy()
                    )

                    loss_mean = np.mean(loss_mean)

                    print(
                        f"Epoch: {i},   AUC: {auc:.4f},   Loss Mean: {loss_mean:.4f}"
                    )
                    
                    # 마지막 에포크에서 시각화 수행
                    if i == num_epochs:
                        # 전체 attention 패턴 시각화
                        att_fig = visualize_global_attention_patterns(self, dataset)
                        att_fig.savefig('attention_heatmap_global_dkt4.png', bbox_inches='tight', dpi=300)
                        
                        # 관계 그래프 시각화
                        graph_fig = visualize_global_relationship_graph(self, dataset)
                        graph_fig.savefig('attention_graph_global_dkt4.png', bbox_inches='tight', dpi=300)
                        
                        # 상세 attention 패턴
                        att_fig1, att_fig2 = visualize_attention_patterns(self, dataset)
                        att_fig1.savefig('attention_heatmap_dkt4.png')
                        att_fig2.savefig('attention_graph_dkt4.png')
                        
                        # 학습 진행 상황
                        prog_fig = plot_learning_progress(self, dataset)
                        prog_fig.savefig('learning_progress_dkt4.png')
                        
                        # 스킬 숙련도 분석
                        skill_fig = analyze_skill_mastery(self, dataset)
                        skill_fig.savefig('skill_mastery_dkt4.png')

# 하이퍼파라미터 설정
batch_size = 256
num_epochs = 50  # 에포크 수 증가 (더 복잡한 데이터셋)
learning_rate = 0.001

# 모델 초기화
model = SAKT(
    num_q=dataset.num_q,
    n=100,          # 시퀀스 길이
    d=256,          # 임베딩 차원 증가 (더 풍부한 특성 표현)
    num_attn_heads=8,  # attention head 증가
    dropout=0.2
).to("cpu")

# 옵티마이저 설정
opt = Adam(model.parameters(), learning_rate)

# 모델 학습
model.train_model(
    train_loader, test_loader, num_epochs, opt
)
