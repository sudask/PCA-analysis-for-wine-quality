import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo 
  
# prepare data
wine_quality = fetch_ucirepo(id=186) 

wine_data = wine_quality.data.original
red_wine_data = wine_data[wine_data['color'] == 'red']
red_wine_data = red_wine_data.drop(columns=['color'])
y = red_wine_data.pop('quality')
X = red_wine_data


# compute correlation matrix
correlation_matrix = X.corr()

# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns, fmt=".2f", annot_kws={"size": 8}, linewidths=0.5)
# plt.xticks(rotation=45)  # 旋转x轴标签
# plt.yticks(rotation=45)  # 旋转y轴标签
# plt.title('Correlation Matrix')
# plt.subplots_adjust(bottom=0.2)
# plt.show()

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(data=X_scaled, columns=X.columns)

pca = PCA(n_components=11) 
pca.fit(X_scaled)

variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(variance_ratio)

df_variance = pd.DataFrame({'Variance Ratio': variance_ratio,
                            'Cumulative Variance Ratio': cumulative_variance_ratio},
                            index=np.arange(1, len(variance_ratio) + 1))

print("各个特征的方差贡献率和累计贡献率：")
print(df_variance)

plt.figure(figsize=(10, 6))
plt.bar(np.arange(1, len(variance_ratio) + 1), variance_ratio, label='Variance Ratio', color='b', alpha=0.7)
plt.plot(np.arange(1, len(variance_ratio) + 1), cumulative_variance_ratio, marker='o', color='r', label='Cumulative Variance Ratio', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance Ratio')
plt.title('Variance Contribution of PCA')
plt.legend()
plt.xticks(np.arange(1, len(variance_ratio) + 1))
plt.grid()
# plt.show()

pca_loadings = pd.DataFrame(pca.components_, columns=X.columns)
print("主成分的载荷矩阵：")
print(pca_loadings)


