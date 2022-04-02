# Feature-Selection

비지도 피처 선택을 다양한 알고리즘을 사용하여 진행하는 프로젝트입니다.

DataSet 폴더에 있는 csv, mat 파일을 사용하여 K-means와 Silhouette Score를 기반으로 FS(Feature Selection)를 진행하게 됩니다.




현재까지 진행한 각 파일 설명입니다.

1~4번 데이터셋은 아래에 설명한 파일을 사용합니다.

1번 데이터셋 : google_review_ratings.csv

2번 데이터셋 : perfume_data.csv

3번 데이터셋 : hcvdat0.csv

4번 데이터셋 : Sales_Transactions_Dataset_weekly.csv




test파일은 YaleB_32x32.mat 데이터셋을 사용하여 진행 중입니다.

K_Means.py : 데이터셋의 시각화를 확인하기 위해 K-means를 사용한 뒤 클러스터를 원하는 피처, 또는 PCA를 사용하여 확인합니다.

SA_K_Means.py : FS에 SA(Simulated Annealing) 알고리즘을 적용한 뒤 시각화하는 파일입니다.

GA_K_Means.py : FS에 GA(Genetic Algorithm)를 적용한 뒤 시각화하는 파일입니다.

std_K_Means.py : 각 피처의 표준편차를 구한 뒤 FS를 적용 후 시각화하는 파일입니다.

Tabu_K_Means.py : FS에 Tabu Search 알고리즘을 적용 후 시각화하는 파일입니다.

모든 파일은 Silhouette Score를 사용하여 FS를 진행합니다. 

[![Hits](https://hits.sh/github.com/dydekdnlt/FS.svg?view=today-total&style=plastic)](https://hits.sh/github.com/dydekdnlt/FS/)
