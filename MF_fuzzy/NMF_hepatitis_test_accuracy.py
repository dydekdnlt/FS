import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
import skfuzzy as fuzz
from scipy.sparse.linalg import svds
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier

df = pd.read_csv('C:\\Users\\ForYou\\Desktop\\DataSet\\hepatitis.csv', header=None)
imputer = SimpleImputer(strategy="mean")
df = pd.DataFrame(imputer.fit_transform(df))

label = np.array(df[0])

print(label)
df = df.drop([0], axis=1)
df_numpy = df.to_numpy()

n_sample, n_feature = df_numpy.shape
A = kneighbors_graph(df_numpy, 3)
B = pairwise_distances(df_numpy)
zero_matrix = np.zeros([n_sample,n_sample])
A = A.toarray()

valueT = df_numpy.T
#print(valueT[0])
t = 1000
for i in range(n_sample):
    for j in range(n_sample):
        if A[i][j] == 1:
            zero_matrix[i][j] = np.exp(-((B[i][j]**2)/t))
            #print(zero_matrix[i][j])

one = np.ones(n_sample)
#print(one)
D = np.diag(zero_matrix @ one.T)
#print(zero_matrix)
#print(D)
L = D - zero_matrix
#print(L)
pd_L = pd.DataFrame(L)
#print(pd_L)

testA = valueT[0] - (valueT[0].T @ D @ one / (one.T @ D @ one)) * one
#print(testA)
testlist = []
for i in range(n_feature):
    testlist.append(valueT[i] - (valueT[i].T @ D @ one / (one.T @ D @ one)) * one)

LS = pd.DataFrame(testlist)
LS = LS.to_numpy()
#print(LS.T.shape)

score_list = []

for i in range(n_feature):
    score_list.append(LS[i].T @ L @ LS[i] / LS[i].T @ D @ LS[i])
#print(score_list)
AA = sorted(score_list)

#print(AA, len(AA))
test_accuracy = []
for x in range(3, n_feature+1):
    select_f = x
    index_list = []
    i_accuracy = []
    cluster = 2
    for i in range(select_f):
        index_list.append(score_list.index(AA[i]))
    #print(index_list)
    selected_features = df_numpy[:, index_list[0:]]
    # print(selected_features)

    kmeans = KMeans(n_clusters=cluster, random_state=0)

    nmf = NMF(n_components=cluster)
    W = nmf.fit_transform(selected_features)
    H = nmf.components_
    #print("W 확인", W, W.shape, sep='\n')
    #print("H 확인", H, H.shape, sep='\n')

    test_nmf = W
    kmeans_test_nmf = kmeans.fit_predict(test_nmf)

    #print(kmeans.labels_)
    for y in range(20):
        new_clf = KNeighborsClassifier(n_neighbors=3)
        new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(df, label, test_size=0.20)
        # new_scores = cross_val_score(new_clf, new_X_train, new_Y_train.ravel(), cv=5)
        #print("new score : ", new_scores)
        #print("new mean accuracy of validation : ", np.mean(new_scores))
        new_clf = new_clf.fit(new_X_train, new_Y_train.ravel())
        new_Y_pred = new_clf.predict(new_X_test)
        #print(new_Y_pred)
        #print(new_Y_test.ravel())
        minScore = accuracy_score(new_Y_test.ravel(), new_Y_pred)
        print("정확도(accuracy) : ", minScore, selected_features.shape)
        #precision = precision_score(new_Y_test.ravel(), new_Y_pred, average='macro')
        #print("정밀도(precision) : ", precision)
        #recall = recall_score(new_Y_test.ravel(), new_Y_pred, average='macro')
        #print("민감도(recall) : ", recall)

        #pca = PCA(n_components=2)
        #pca_tra = pca.fit_transform(selected_features)
        #print("time : ", time.time() - start)
        #sns.scatterplot(x=pca_tra[:,0], y=pca_tra[:, 1], c=label)
        # plt.show()
        i_accuracy.append(minScore)
    test_accuracy.append(sum(i_accuracy)/len(i_accuracy))
    # print(i_accuracy, len(i_accuracy))
print(test_accuracy, len(test_accuracy))