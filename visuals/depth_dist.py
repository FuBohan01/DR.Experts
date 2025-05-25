import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from glob import glob
import cv2
from scipy.spatial.distance import cdist

# 假设所有深度图像都存储在某个文件夹内
data_folder = (
    "/mnt/iMVR/guanyi/dataset/IQA/ChallengeDB_release/Depths"  # 请替换为你的数据集路径
)
image_paths = glob(os.path.join(data_folder, "*-depth.png"))  # 假设深度图像格式是 PNG
mean = 0.485
std = 0.229
# 读取第一张图像获取尺寸信息
sample_image = (cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE) / 255) - mean / std
img_shape = sample_image.shape

# 读取所有图像并展平
depth_images = []
for path in image_paths:
    img = (cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255) - mean / std
    if img is None or img.shape != img_shape:
        continue  # 跳过无法读取或尺寸不匹配的图像
    depth_images.append(img.flatten())

# 转换为 NumPy 数组
depth_images = np.array(depth_images)

# 随机选择 100 张图像索引
random_indices = np.random.choice(len(depth_images), 300, replace=False)

# 获取 100 张随机抽样的图像
depth_images = depth_images[random_indices]

# # 进行 PCA 降维到 3D
# pca = TruncatedSVD(n_components=6)
# pca_result = pca.fit_transform(depth_images)

# principal_components = pca.components_  # (3, flattened image size)

# # 重新调整回图像形状
# pc1_image = principal_components[0].reshape(img_shape)
# pc2_image = principal_components[1].reshape(img_shape)
# pc3_image = principal_components[2].reshape(img_shape)


# # 归一化到 0-255 方便可视化
# def normalize_and_save(image, filename):
#     norm_img = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
#     norm_img = norm_img.astype(np.uint8)
#     cv2.imwrite(filename, norm_img)


# # 保存 PCA 主成分图像
# normalize_and_save(pc1_image, "pca_component_1.png")
# normalize_and_save(pc2_image, "pca_component_2.png")
# normalize_and_save(pc3_image, "pca_component_3.png")

# print(
#     "✅ 三张主成分图像已保存为 'pca_component_1.png', 'pca_component_2.png', 'pca_component_3.png'！"
# )


# # 获取 PCA 解释方差比例
# explained_variance_ratio = pca.explained_variance_ratio_
# cumulative_variance = np.cumsum(explained_variance_ratio)

# # 打印出每个主成分的贡献率
# print("🔹 PCA 主成分信息量贡献度（解释方差比）:")
# for i, ratio in enumerate(explained_variance_ratio):
#     print(f"  - 主成分 {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

# # 取绝对值
# pca_result_abs = np.abs(pca_result)

# # 使用 K-Means 进行聚类，尝试 3 到 10 组，寻找最佳 k
# inertia = []
# k_range = range(3, 21)

# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(pca_result_abs)
#     inertia.append(kmeans.inertia_)

# # 画出肘部法则曲线，确定最佳 k
# plt.figure(figsize=(8, 5))
# plt.plot(k_range, inertia, marker="o", linestyle="-")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Inertia")
# plt.title("Elbow Method for Optimal k")
# plt.savefig("elbow_method.png")  # 保存肘部法则图像
# plt.close()  # 设定 K-Means 聚类中心数
# optimal_k = 4
# kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
# labels = kmeans.fit_predict(pca_result_abs)
# centers = kmeans.cluster_centers_

# # 计算每个样本到其所属聚类中心的欧式距离
# distances = cdist(pca_result_abs, centers, metric="euclidean")

# # 选取最靠近聚类中心的样本
# num_samples_per_cluster = 10  # 每个聚类中心展示4张原始深度图
# closest_samples = {}

# for cluster in range(optimal_k):
#     # 获取当前类别的样本索引
#     cluster_indices = np.where(labels == cluster)[0]
#     # 按距离排序，找到最靠近聚类中心的 num_samples_per_cluster 张图片
#     closest_indices = cluster_indices[
#         np.argsort(distances[cluster_indices, cluster])[:num_samples_per_cluster]
#     ]
#     closest_samples[cluster] = closest_indices

# # 创建图像网格展示聚类中心及其样本
# fig, axes = plt.subplots(optimal_k, num_samples_per_cluster + 1, figsize=(12, 8))

# for cluster in range(optimal_k):
#     # 可视化聚类中心（PCA 近似重建，使用聚类中心数据）
#     # ax = axes[cluster, 0]
#     # cluster_image = centers[cluster].reshape(img_shape)  # 将 PCA 还原回原始图像形状
#     # cluster_image = (
#     #     (cluster_image - np.min(cluster_image))
#     #     / (np.max(cluster_image) - np.min(cluster_image))
#     #     * 255
#     # )
#     # cluster_image = cluster_image.astype(np.uint8)  # 归一化到 0-255
#     # ax.imshow(cluster_image, cmap="gray")
#     # ax.set_title(f"Cluster {cluster}\nCenter")
#     # ax.axis("off")

#     # 显示最靠近该中心的原始深度图样本
#     for i, sample_idx in enumerate(closest_samples[cluster]):
#         ax = axes[cluster, i + 1]
#         sample_image = depth_images[sample_idx].reshape(img_shape)
#         ax.imshow(sample_image, cmap="gray")
#         ax.set_title(f"Sample {i+1}")
#         ax.axis("off")

# # 调整布局并保存
# plt.tight_layout()
# plt.savefig("cluster_samples.png", dpi=300)
# plt.close()

# print("✅ 聚类中心及其样本已保存到 'cluster_samples.png'！请下载查看。")


# Step 1: 进行 t-SNE 降维到 2D
print("🔹 正在使用 t-SNE 降维到 2D...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
tsne_result = tsne.fit_transform(depth_images)  # t-SNE 处理 PCA 特征
# # # 取绝对值
# tsne_result = np.abs(tsne_result)

# Step 2: 在 t-SNE 空间中进行 K-Means 聚类
optimal_k = 7  # 设定4个聚类
print(f"🔹 正在使用 K-Means 聚类 (k={optimal_k})...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(tsne_result)
centers = kmeans.cluster_centers_  # 获取聚类中心

# Step 3: 计算每个样本到其所属聚类中心的距离，选取最靠近聚类中心的样本
distances = cdist(tsne_result, centers, metric="euclidean")
num_samples_per_cluster = 4  # 每个聚类中心展示4张原始深度图
closest_samples = {}

for cluster in range(optimal_k):
    # 获取当前类别的样本索引
    cluster_indices = np.where(labels == cluster)[0]
    # 按距离排序，找到最靠近聚类中心的 num_samples_per_cluster 张图片
    closest_indices = cluster_indices[
        np.argsort(distances[cluster_indices, cluster])[:num_samples_per_cluster]
    ]
    closest_samples[cluster] = closest_indices

# Step 4: 可视化 t-SNE 聚类结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="tab10", alpha=0.8
)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    label="Cluster Centers",
)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization with K-Means Clustering")
plt.legend()
plt.savefig("tsne_clusters.png", dpi=300)
plt.close()
print("✅ t-SNE 聚类图已保存为 'tsne_clusters.png'")

# Step 5: 可视化每个聚类中心的样本
fig, axes = plt.subplots(optimal_k, num_samples_per_cluster + 1, figsize=(12, 8))

for cluster in range(optimal_k):
    # 可视化 t-SNE 聚类中心（无法反变换到原始图像，只是中心点）
    ax = axes[cluster, 0]
    ax.text(
        0.5, 0.5, f"Cluster {cluster}\nCenter", fontsize=12, ha="center", va="center"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # 显示最靠近该中心的原始深度图样本
    for i, sample_idx in enumerate(closest_samples[cluster]):
        ax = axes[cluster, i + 1]
        sample_image = depth_images[sample_idx].reshape(img_shape)
        ax.imshow(sample_image, cmap="gray")
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")

# 调整布局并保存
plt.tight_layout()
plt.savefig("tsne_cluster_samples.png", dpi=300)
plt.close()
# from scipy.ndimage import gaussian_filter

# # print("✅ t-SNE 聚类中心及其样本已保存为 'tsne_cluster_samples.png'！请下载查看。")
# # Step 1: 运行 t-SNE 降维到 2D
# print("🔹 正在使用 t-SNE 降维到 2D...")
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
# tsne_result = tsne.fit_transform(depth_images)  # t-SNE 处理 PCA 特征

# # Step 2: 计算 2D 直方图密度分布
# print("🔹 计算 2D 密度分布...")
# x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
# y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()

# # 创建二维直方图
# grid_size = 100  # 设定网格大小
# x_bins = np.linspace(x_min, x_max, grid_size)
# y_bins = np.linspace(y_min, y_max, grid_size)
# density, _, _ = np.histogram2d(
#     tsne_result[:, 0], tsne_result[:, 1], bins=[x_bins, y_bins]
# )

# # Step 3: 进行二维高斯平滑
# print("🔹 进行 2D 高斯平滑处理...")
# sigma = 3.5  # 平滑程度
# smoothed_density = gaussian_filter(density, sigma=sigma)

# # Step 4: 进行对数变换增强对比度
# print("🔹 进行对比度增强 (log + exp)...")
# log_density = np.log1p(smoothed_density)  # log(1 + x) 变换
# exp_density = np.exp(smoothed_density) - 1  # e^x - 1 变换（可选）

# # Step 5: 可视化平滑后的概率分布图（对比 log vs exp）
# fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# # 对数变换可视化
# ax1 = axes[0]
# img1 = ax1.imshow(
#     log_density.T, origin="lower", cmap="viridis", extent=[x_min, x_max, y_min, y_max]
# )
# fig.colorbar(img1, ax=ax1, label="Log Density")
# ax1.set_title("t-SNE 2D Gaussian-Smoothed Log Density")
# ax1.set_xlabel("t-SNE Component 1")
# ax1.set_ylabel("t-SNE Component 2")

# # 指数变换可视化
# ax2 = axes[1]
# img2 = ax2.imshow(
#     exp_density.T, origin="lower", cmap="viridis", extent=[x_min, x_max, y_min, y_max]
# )
# fig.colorbar(img2, ax=ax2, label="Exp Density")
# ax2.set_title("t-SNE 2D Gaussian-Smoothed Exp Density")
# ax2.set_xlabel("t-SNE Component 1")
# ax2.set_ylabel("t-SNE Component 2")

# # 保存增强后的图像
# plt.tight_layout()
# plt.savefig("tsne_gaussian_enhanced.png", dpi=300)
# plt.close()

# print("✅ t-SNE 高斯平滑后的对比度增强图已保存为 'tsne_gaussian_enhanced.png'！")
