from math import sqrt
from os import listdir, path
import os
import shutil
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64

model_folder = "./model"

def vec_len(v):
    sum = 0
    for component in v:
        sum += component ** 2
    return sqrt(sum)

def load_dataset(filename):
    #csv 데이터 로드
    df = pd.read_csv(filename, index_col=0, usecols=range(4))

    #시간 데이터 소숫점 버리기
    df.index = df.index.map(lambda x: int(x))

    #10초까지 자르기
    df = df.loc[df.index < 10]

    #합력 계산
    df = df.apply(lambda d: vec_len(d), axis="columns", result_type="expand")

    #평균 groupby
    average = df.groupby(df.index.name).mean()

    #numpy 배열로 변환
    return np.array(average)

def load_datasets_in_folder(folder):
    datasets = np.empty((0, 10))
    file_names = []
    for f in listdir(folder):
        full_path = path.join(folder, f)
        if not path.isfile(full_path) or path.splitext(full_path)[1] != ".csv":
            continue
        dataset = load_dataset(full_path)
        if len(dataset) != 10:
            print(f'파일 로드 오류: {full_path}, dataset의 크기가 10이 아니고 {len(dataset)} 입니다. 무시합니다.')
            continue
        print(f'데이터 로드 성공: {full_path}')
        datasets = np.vstack((datasets, dataset))
        file_names.append(f)
    return (datasets, file_names)


def load_training_data():
    #데이터셋 정의
    X = np.empty((0, 10))
    Y = np.empty((1, 0))
    #뜀 데이터 로드
    print("뜀 데이터를 로드합니다.")
    run_data = load_datasets_in_folder("./dataset_run")[0]
    X = np.vstack((X, run_data))
    Y = np.append(Y, np.repeat(1, len(run_data)))
    #걸음 데이터 로드
    print("걸음 데이터를 로드합니다.")
    walk_data = load_datasets_in_folder("./dataset_walk")[0]
    X = np.vstack((X, walk_data))
    Y = np.append(Y, np.repeat(0, len(walk_data)))
    Y = np.reshape(Y, (len(Y), 1))
    return (X, Y)

# 입력할 데이터
# X = np.array([[1.029815, 1.51113, 1.890595, 2.36834, 2.395795, 2.003316583, 2.18814, 1.843235, 1.59085, 1.21185],
#               [0.029815, 0.51113, 0.890595, 0.36834, 0.395795, 0.003316583, 0.18814, 0.843235, 0.59085, 0.21185]])
# Y = np.array([[0], [1]])

#StandardScaler 설정
scaler = StandardScaler()

#csv로부터 데이터 로드
X, Y = load_training_data()

#전처리
scaler = scaler.fit(X)
X = scaler.transform(X)
# X = np.round(X,2)

print(X)
print(Y)

input_size = 10
hidden_size_1 = 20
hidden_size_2 = 20
hidden_size_3 = 20

output_size = 1
rng = Generator(PCG64(seed=425028234))
# 가중치 랜덤 1, 2, 3
W1 = rng.random((input_size, hidden_size_1))

W2 = rng.random((hidden_size_1, hidden_size_2))

W3 = rng.random((hidden_size_2, hidden_size_3))

W4 = rng.random((hidden_size_3, output_size))

# W1 = np.array([[0.1,0.2],
#                [0.3,0.4]])

# W2 = np.array([[0.5,0.6],``
#                [0.7,0.8]])

# W3 = np.array([[0.9],
#                [0.95]])

# print("---------------------------------")
# print(W1)
# print("---------------------------------")
# print(W2)
# print("---------------------------------")
# print(W3)
# print("---------------------------------")

# 학습률과 반복수
learning_rate = 0.01
num_iterations = 500000
error_list = []
error_10000 = []
# error_list = []
# 학습시작
for i in range(num_iterations):
    # 순전파
    z1 = np.dot(X, W1)
    # print(z1)
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(a1, W2)
    a2 = 1 / (1 + np.exp(-z2))
    z3 = np.dot(a2, W3)
    a3 = 1 / (1 + np.exp(-z3))
    z4 = np.dot(a3, W4)
    y_hat = 1 / (1 + np.exp(-z4))

    #역전파
    error = Y - y_hat
    
    delta_output = error * y_hat * (1 - y_hat)
    delta_hidden_3 = delta_output.dot(W4.T) * a3 * (1 - a3)
    delta_hidden_2 = delta_hidden_3.dot(W3.T) * a2 * (1 - a2)
    delta_hidden_1 = delta_hidden_2.dot(W2.T) * a1 * (1 - a1)
    W4 += learning_rate * a3.T.dot(delta_output)
    W3 += learning_rate * a2.T.dot(delta_hidden_3)
    W2 += learning_rate * a1.T.dot(delta_hidden_2)
    W1 += learning_rate * X.T.dot(delta_hidden_1)

    if i % 100000 == 0:
        print(i,"회 반복 : ",error)
        error_list.append(abs(error))

error_10000.append(error_list)
error_10000 = np.array(error_10000)
error_10000 = abs(error_10000.flatten())
plt.yticks(np.arange(0, 1.1, 0.01))
plt.plot(error_10000, label='error')
plt.legend()
plt.show()

if path.exists(model_folder):
    print("이미 존재하는 네트워크를 삭제합니다.")
    shutil.rmtree(model_folder)
os.makedirs(model_folder, exist_ok=True)

print(f"학습된 네트워크를 다음 위치에 저장합니다: {model_folder}")

joblib.dump(scaler, path.join(model_folder, "scaler.joblib"))

np.savez(path.join(model_folder, 'model.npz'), *[W1, W2, W3, W4])