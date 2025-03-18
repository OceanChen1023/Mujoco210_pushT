import numpy as np
import zarr
import os

# 設定 Zarr 資料夾
zarr_directory = "Demo"
output_zarr_path = f"{zarr_directory}/replay_buffer.zarr"

# 找出所有 .zarr 檔案
zarr_files = sorted([f for f in os.listdir(zarr_directory) if f.endswith(".zarr")])

if len(zarr_files) == 0:
    raise ValueError("❌ 沒有找到任何 .zarr 檔案！請確認資料夾內是否有 Zarr 檔案。")

# 開啟第一個 Zarr 作為基礎
base_zarr_path = os.path.join(zarr_directory, zarr_files[0])
print("base_zarr_path: " + base_zarr_path)
base_store = zarr.open(base_zarr_path, mode="r")
print("base store tree:", base_store.tree())

# 獲取所有鍵值
data_keys = set(base_store["data"].array_keys())

# ✅ 確保 "meta" 存在
meta_keys = set(base_store["meta"].keys()) if "meta" in base_store else set()

if "meta" in base_store:
    print("Meta keys:", base_store["meta"].keys())
else:
    print("Meta group does not exist!")

# 建立新的合併 Zarr 檔案
zarr_out = zarr.open(output_zarr_path, mode="w")
zarr_out.create_group("data")
zarr_out.create_group("meta")

# 初始化合併儲存空間
merged_data = {key: base_store["data"][key][:] for key in data_keys}
merged_meta = {key: base_store["meta"][key][:] for key in meta_keys} if meta_keys else {}

# ✅ 初始化 Meta 累積值
meta_accumulated = {key: base_store["meta"][key][:] for key in meta_keys} if meta_keys else {}

# 遍歷其他 `.zarr` 檔案並合併
for zarr_file in zarr_files[1:]:
    zarr_path = os.path.join(zarr_directory, zarr_file)
    print(f"🔄 合併 {zarr_file} ...")

    store = zarr.open(zarr_path, mode="r")

    # ✅ 確保結構匹配
    if set(store["data"].array_keys()) != data_keys:
        raise ValueError(f"❌ {zarr_file} 的 data 結構不匹配！")
    if "meta" in store and set(store["meta"].keys()) != meta_keys:
        raise ValueError(f"❌ {zarr_file} 的 meta 結構不匹配！")

    # ✅ 合併 Data
    for key in data_keys:
        merged_data[key] = np.concatenate([merged_data[key], store["data"][key][:]])

    # ✅ 合併 Meta（如果有）
    if "meta" in store:
        for key in meta_keys:
            if key not in store["meta"]:
                print(f"⚠️ Warning: '{key}' not found in {zarr_file}, skipping!")
                continue
            
            current_meta = store["meta"][key][:]

            # ✅ 確保有東西才累加
            if len(meta_accumulated[key]) > 0:
                accumulated_meta = current_meta + meta_accumulated[key][-1]
            else:
                accumulated_meta = current_meta

            merged_meta[key] = np.concatenate([merged_meta[key], accumulated_meta])
            meta_accumulated[key] = accumulated_meta

# ✅ 儲存合併後的數據
for key in data_keys:
    zarr_out["data"].create_dataset(key, data=merged_data[key], overwrite=True)

if meta_keys:
    for key in meta_keys:
        zarr_out["meta"].create_dataset(key, data=merged_meta[key], overwrite=True)

print(f"✅ 合併完成，結果已儲存於 {output_zarr_path}！")
