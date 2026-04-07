from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

data = pd.read_csv("data.csv")

points = data[["x3", "x7"]].to_numpy()


def run_dbscan(arr: np.ndarray, radius: float, min_pts: int) -> np.ndarray:
    n = arr.shape[0]

    UNSEEN = -2

    result = np.full(n, UNSEEN, dtype=int)

    kd = cKDTree(arr)

    neighbors_list = kd.query_ball_point(arr, r=radius)

    current_cluster = 0

    for i in range(n):
        if result[i] != UNSEEN:
            continue

        neighbors = neighbors_list[i]

        if len(neighbors) < min_pts:
            result[i] = -1
            continue

        # старт нового кластера
        queue = deque(neighbors)
        result[i] = current_cluster

        while queue:
            point_id = queue.popleft()

            if result[point_id] == -1:
                result[point_id] = current_cluster

            if result[point_id] != UNSEEN:
                continue

            result[point_id] = current_cluster
            local_neighbors = neighbors_list[point_id]

            if len(local_neighbors) >= min_pts:
                queue.extend(local_neighbors)

        current_cluster += 1

    return result


RADIUS = 10
MIN_POINTS = 15

print(f"Старт кластеризации (radius={RADIUS}, min_pts={MIN_POINTS})")

clusters = run_dbscan(points, RADIUS, MIN_POINTS)

unique_clusters = set(clusters)
cluster_count = len(unique_clusters) - (1 if -1 in clusters else 0)
noise_count = np.sum(clusters == -1)

print("Кластеры:", cluster_count)
print("Шум:", noise_count)

data_out = data.copy()
data_out["group"] = clusters

x_vals = data_out["x3"]
y_vals = -data_out["x7"]

plt.figure(figsize=(16, 6))

# шум
mask_noise = clusters == -1
plt.scatter(x_vals[mask_noise], y_vals[mask_noise],
            c="lightgray", s=1, alpha=0.3)

# кластеры
mask_clusters = clusters != -1
plt.scatter(x_vals[mask_clusters], y_vals[mask_clusters],
            c=clusters[mask_clusters], cmap="tab20", s=2)

plt.title("DBSCAN result")
plt.xlabel("x3")
plt.ylabel("-x7")

plt.axis("equal")
plt.tight_layout()

plt.savefig("result.png", dpi=200)
plt.show()

data_out.to_csv("result_dbscan.csv", index=False)

print("Готово")
