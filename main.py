import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. CSV 파일을 절대경로로 불러오기 (아래 경로는 예시입니다)
df = pd.read_csv("C:/Users/jumiiilin/Desktop/Delivery.csv")

# 2. 위도, 경도 추출
coords = df[['Latitude', 'Longitude']]

# 3. K-Means 클러스터링 (4개 클러스터)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(coords)

# 4. 클러스터 중심 좌표 저장
centroids = kmeans.cluster_centers_

# 5. 시각화
plt.figure(figsize=(10, 8))
for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'Cluster {cluster}')
    
# 중심점 시각화
plt.scatter(centroids[:, 1], centroids[:, 0], c='black', s=200, marker='X', label='Centroids')

# 그래프 꾸미기
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-Means Clustering of Delivery Points')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

