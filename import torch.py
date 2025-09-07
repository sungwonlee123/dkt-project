

import torch
import torch.nn as nn
import pandas as pd
import os
import requests
import zipfile

# DKT 모델 클래스 정의 (가장 상단에 위치)
class DKT(nn.Module):
    def __init__(self, num_questions, hidden_size):
        super(DKT, self).__init__()
        self.embedding = nn.Embedding(num_questions * 2, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_questions)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return torch.sigmoid(out)

# ASSISTments 2009 데이터셋 다운로드 및 로딩 예시
assist_path = "skill_builder_data.csv"
assist_df = pd.read_csv(assist_path, encoding='latin1')
print("ASSISTments 샘플:", assist_df.head())

# ASSISTments 데이터셋 → DKT 입력 변환 예시
# (user_id, skill_id, correct) 기준으로 시퀀스 생성
assist_sample = assist_df[['user_id', 'skill_id', 'correct']].dropna().head(10)
num_skills = assist_df['skill_id'].nunique()
def make_dkt_sequence(df, num_skills):
    # 문제 번호: skill_id, 정답 여부: correct
    seq = (df['skill_id'].astype(int) * 2 + df['correct'].astype(int)).tolist()
    targets = df['correct'].astype(int).tolist()
    return seq, targets
assist_seq, assist_targets = make_dkt_sequence(assist_sample, num_skills)

# DKT 모델 실습: ASSISTments 시퀀스 사용
num_questions = num_skills
hidden_size = 16
seq_len = len(assist_seq)
inputs = torch.tensor([assist_seq], dtype=torch.long)  # (batch=1, seq_len)
targets = torch.tensor([assist_targets], dtype=torch.float32)  # (batch=1, seq_len)

model = DKT(num_questions=num_questions, hidden_size=hidden_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
optimizer.zero_grad()
outputs = model(inputs)
pred = outputs[:, :, 0] if outputs.shape[2] > 1 else outputs.squeeze(-1)
loss = criterion(pred, targets)
loss.backward()
optimizer.step()
print(f"ASSISTments 학습 Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    outputs = model(inputs)
    pred = (outputs[:, :, 0] if outputs.shape[2] > 1 else outputs.squeeze(-1)) > 0.5
    print("ASSISTments 예측 결과:", pred)



# kt1 폴더 내 모든 CSV 파일을 하나로 병합 (최초 1회만)
import glob
ednet_merged_path = "ednet_merged.csv"
if os.path.exists(ednet_merged_path):
    ednet_df = pd.read_csv(ednet_merged_path)
    print("EdNet 병합 파일 로드 샘플:", ednet_df.head())
else:
    kt1_folder = "KT1"
    csv_files = glob.glob(os.path.join(kt1_folder, "*.csv"))
    print(f"kt1 폴더 내 CSV 파일 개수: {len(csv_files)}")
    df_list = [pd.read_csv(f) for f in csv_files]
    ednet_merged = pd.concat(df_list, ignore_index=True)
    ednet_merged.to_csv(ednet_merged_path, index=False)
    print("EdNet 병합 샘플:", ednet_merged.head())
    ednet_df = ednet_merged
# EdNet 데이터셋 → DKT 입력 변환 예시
ednet_sample = ednet_df[['user_id', 'concept_id', 'correct']].dropna().head(10)
num_concepts = ednet_df['concept_id'].nunique()
ednet_seq = (ednet_sample['concept_id'].astype(int) * 2 + ednet_sample['correct'].astype(int)).tolist()
ednet_targets = ednet_sample['correct'].astype(int).tolist()
# DKT 모델 실습: EdNet 시퀀스 사용
num_questions = num_concepts
seq_len = len(ednet_seq)
inputs = torch.tensor([ednet_seq], dtype=torch.long)
targets = torch.tensor([ednet_targets], dtype=torch.float32)

model = DKT(num_questions=num_questions, hidden_size=hidden_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
optimizer.zero_grad()
outputs = model(inputs)
pred = outputs[:, :, 0] if outputs.shape[2] > 1 else outputs.squeeze(-1)
loss = criterion(pred, targets)
loss.backward()
optimizer.step()
print(f"EdNet 학습 Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    outputs = model(inputs)
    pred = (outputs[:, :, 0] if outputs.shape[2] > 1 else outputs.squeeze(-1)) > 0.5
    print("EdNet 예측 결과:", pred)
   