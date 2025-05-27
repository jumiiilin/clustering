import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 데이터 불러오기
df = pd.read_csv("Delivery.csv")

# 클러스터링에 사용할 컬럼 추출
coords = df[['Latitude', 'Longitude']]

# KMeans 모델 생성 및 학습 (클러스터 수는 4로 설정, 필요 시 변경 가능)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(coords)

# 클러스터 중심점 추출
centroids = kmeans.cluster_centers_

# 결과 시각화
plt.figure(figsize=(10, 8))
for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'Cluster {cluster}')
plt.scatter(centroids[:, 1], centroids[:, 0], c='black', s=200, marker='X', label='Centroids')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-Means Clustering of Delivery Points')
plt.legend()
plt.grid(True)
plt.show()

