import pickle
import numpy as np
import os
import sys
import pandas as pd

# 兼容旧版pkl的模块路径映射
sys.modules['pandas.indexes'] = pd.core.indexes
sys.modules['pandas.indexes.base'] = pd.core.indexes.base


def main():
    pkl_path = "./data/LSWMD.pkl"
    print("Loading raw pkl data...")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    x_list = []
    y_list = []
    label_names = []

    for idx, row in data.iterrows():
        wafer_map = row['waferMap']
        defect_type = row['failureType']

        # 跳过无标注样本
        if len(defect_type) == 0:
            continue

        label = defect_type[0]
        if label not in label_names:
            label_names.append(label)

        h, w = wafer_map.shape
        if h != 26 or w != 26:
            padded = np.zeros((26, 26), dtype=np.float32)
            start_h = max(0, (26 - h) // 2)
            start_w = max(0, (26 - w) // 2)
            end_h = min(26, start_h + h)
            end_w = min(26, start_w + w)
            padded[start_h:end_h, start_w:end_w] = wafer_map[:end_h-start_h, :end_w-start_w]
            wafer_map = padded

        x_list.append(wafer_map.astype(np.float32))
        y_list.append(label_names.index(label))

    x = np.array(x_list)
    y = np.array(y_list)

    # 归一化到[0, 1]区间
    x = x / 2.0
    x = np.expand_dims(x, axis=1)  # 增加通道维度 [N, 1, 26, 26]

    os.makedirs('./data', exist_ok=True)
    np.save('./data/wafer_x.npy', x)
    np.save('./data/wafer_y.npy', y)
    np.save('./data/label_names.npy', np.array(label_names))

    print(f"Done! Total samples: {len(x)}")
    print(f"Number of classes: {len(label_names)}")
    print(f"Class names: {label_names}")
    print("Saved to ./data/wafer_x.npy, wafer_y.npy, label_names.npy")


if __name__ == "__main__":
    main()
