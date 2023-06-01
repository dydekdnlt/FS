import numpy as np
from PIL import Image
import os
import pickle


class SVD:
    def get(self, A, compRate=0.5):
        n = A.shape[0]
        p = A.shape[1]

        U, s, VT = np.linalg.svd(A, full_matrices=False)
        k = int(compRate * (n * p) / (n + 1 + p))
        S = np.diag(s[:k])
        B = np.dot(U[:, :k], np.dot(S, VT[:k, :]))
        B = (255 * (B - np.min(B)) / np.ptp(B)).astype(np.uint8)
        return B, U[:, :k], s[:k], VT[:k, :], k


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


with open('./cifar-10-batches-py/batches.meta', 'rb') as infile:
    data = pickle.load(infile, encoding='latin1')
    classes = data['label_names']

# 클래스 별 폴더 생성
os.mkdir('./train_svd')
os.mkdir('./test_svd')
for name in classes:
    os.mkdir('./train_svd/{}'.format(name))
    os.mkdir('./test_svd/{}'.format(name))
# Trainset Unpacking
# data_batch 파일들 순서대로 unpacking
for i in range(1, 6):
    print('Unpacking Train File {}/{}'.format(i, 5))
    train_file = unpickle('./cifar-10-batches-py/data_batch_{}'.format(i))

    train_data = train_file[b'data']

    # 10000, 3072 -> 10000, 3, 32, 32 형태로 변환
    train_data_reshape = np.vstack(train_data).reshape((-1, 3, 32, 32))
    # 이미지 저장을 위해 10000, 32, 32, 3으로 변환
    train_data_reshape = train_data_reshape.swapaxes(1, 3)
    train_data_reshape = train_data_reshape.swapaxes(1, 2)
    # 레이블 리스트 생성
    train_labels = train_file[b'labels']
    # 파일 이름 리스트 생성
    train_filename = train_file[b'filenames']
    model = SVD()
    # 10000개의 파일을 순차적으로 저장
    for idx in range(10000):
        train_label = train_labels[idx]
        train_image = Image.fromarray(train_data_reshape[idx])
        print(train_image)
        # 클래스 별 폴더에 파일 저장
        R, _, _, _, _ = model.get(train_image[:, :, 0], 0.1)
        G, _, _, _, _ = model.get(train_image[:, :, 1], 0.1)
        B, _, _, _, _ = model.get(train_image[:, :, 2], 0.1)
        newImg = np.zeros_like(train_image)
        newImg[:, :, 0] = R
        newImg[:, :, 1] = G
        newImg[:, :, 2] = B
        newImg.save('./train_svd/{}/{}'.format(classes[train_label], train_filename[idx].decode('utf8')))
# -----------------------------------------------------------------------------------------
# Testset Unpacking
print('Unpacking Test File')
test_file = unpickle('./cifar-10-batches-py/test_batch')

test_data = test_file[b'data']

# 10000, 3072 -> 10000, 3, 32, 32 형태로 변환
test_data_reshape = np.vstack(test_data).reshape((-1, 3, 32, 32))
# 이미지 저장을 위해 10000, 32, 32, 3으로 변환
test_data_reshape = test_data_reshape.swapaxes(1, 3)
test_data_reshape = test_data_reshape.swapaxes(1, 2)
# 레이블 리스트 생성
test_labels = test_file[b'labels']
# 파일 이름 리스트 생성
test_filename = test_file[b'filenames']
model = SVD()
# 10000개의 파일을 순차적으로 저장
for idx in range(10000):
    test_label = test_labels[idx]
    test_image = Image.fromarray(test_data_reshape[idx])
    # 클래스 별 폴더에 파일 저장
    R, _, _, _, _ = model.get(test_image[:, :, 0], 0.1)
    G, _, _, _, _ = model.get(test_image[:, :, 1], 0.1)
    B, _, _, _, _ = model.get(test_image[:, :, 2], 0.1)
    newImg = np.zeros_like(test_image)
    newImg[:, :, 0] = R
    newImg[:, :, 1] = G
    newImg[:, :, 2] = B
    newImg.save('./test_svd/{}/{}'.format(classes[test_label], test_filename[idx].decode('utf8')))

print('Unpacking Finish')




