
import numpy as np
import PIL.Image as pilimg
import cv2
import os

# n*p 행렬을 SVD로 compRate 만큼 압축한 결과와
# 좌특이행렬, 고유값, 우특이 행렬을 리턴한다.
class SVD:
    def get(self, A, compRate = 0.1):
        n = A.shape[0]
        p = A.shape[1]
# A = U(n,n) * diag(s(n)) * V(n,p)
# full_matrices=False 옵션에 의해 "0"인 부분을 삭제하고 리턴
        U, s, VT = np.linalg.svd(A, full_matrices=False)
        k = int(28 * compRate)
        # k = int(compRate * (n*p) / (n+1+p)) # k: 고유값 사용갯수
        S = np.diag(s[:k])
        B = np.dot(U[:, :k], np.dot(S, VT[:k, :]))
        B = (255*(B - np.min(B))/np.ptp(B)).astype(np.uint8)
        return B, U[:, :k], s[:k], VT[:k, :], k


f = os.listdir('./new_mnist_training_svd_10')
print(f)
pixel_average = 0
for i in f:
    f2 = os.listdir('./new_mnist_training_svd_10/{}'.format(i))
    for j in f2:
        im = cv2.imread("./new_mnist_training_svd_10/{}/{}".format(i, j))
        # print(im)

        model = SVD()
        # 컬러이미지 RGB 차원에 대해 각각 10% 크기로 압축한 결과를 얻는다.
        R, _, _, _, _ = model.get(im[:,:,0], 0.1)
        G, _, _, _, _ = model.get(im[:,:,1], 0.1)
        B, _, _, _, _ = model.get(im[:,:,2], 0.1)

        newImg = np.zeros_like(im)
        newImg[:,:,0] = R
        newImg[:,:,1] = G
        newImg[:,:,2] = B
        '''
        test1 = ((im - newImg) ** 2) ** 0.5
        test1_sum = test1.sum() / 3 / 1024
        test1_aver = round(test1_sum, 2)

        pixel_average += test1_aver
        pixel_average = round(pixel_average, 2)
        print(pixel_average)
        '''
        cv2.imwrite('./new_mnist_training_svd_10/{}/svd_10_{}'.format(i, j), newImg)

# print("final : ", pixel_average / 50000)
'''
f = os.listdir('./train_mini_imagenet')
print(f)

for i in f:
    f2 = os.listdir('./train_mini_imagenet/{}'.format(i))
    for j in f2:
        im = cv2.imread("./train_mini_imagenet/{}/{}".format(i, j))
        # print(im)

        model = SVD()
        # 컬러이미지 RGB 차원에 대해 각각 10% 크기로 압축한 결과를 얻는다.
        R, _, _, _, _ = model.get(im[:,:,0], 0.9)
        G, _, _, _, _ = model.get(im[:,:,1], 0.9)
        B, _, _, _, _ = model.get(im[:,:,2], 0.9)

        newImg = np.zeros_like(im)
        newImg[:,:,0] = R
        newImg[:,:,1] = G
        newImg[:,:,2] = B
        cv2.imwrite('./train_mini_imagenet/{}/svd_90_{}'.format(i, j), newImg)
'''


