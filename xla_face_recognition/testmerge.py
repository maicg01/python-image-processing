import numpy as np

src_names = np.load("/home/maicg/Documents/Me/CERBERUS/GitLab/cccdserverdjango/database/collection_face.npy", allow_pickle=True).tolist()
src = np.load("/home/maicg/Documents/Me/CERBERUS/GitLab/cccdserverdjango/database/collection_face_label.npy", allow_pickle=True).tolist()

merged_data = np.concatenate((src_names, src), axis=0) #Where axis=1 means that we want to concatenate horizontally -> chieu ngang

print(merged_data)


import faiss

def merge_indexes(index1, index2):
    if not isinstance(index1, faiss.IndexFlat) or not isinstance(index2, faiss.IndexFlat):
        raise ValueError("Indexes must be of type IndexFlat")

    # Gộp hai mảng dữ liệu
    merged_data = np.concatenate([np.array(index1.reconstruct(i).reshape((1,512))) for i in range(index1.ntotal)] +
                                 [np.array(index2.reconstruct(i).reshape((1,512))) for i in range(index2.ntotal)])

    # Tạo index mới với mảng dữ liệu gộp
    merged_index = faiss.IndexFlatIP(512)
    merged_index.add(merged_data)

    return merged_index

# Ví dụ sử dụng
index_path_1 = '/home/maicg/Documents/Me/CERBERUS/GitLab/cccdserverdjango/database/cerberus_ada_r50_DB.bin'  # Đường dẫn đến index 1
index_path_2 = '/home/maicg/Documents/Me/CERBERUS/GitLab/cccdserverdjango/database/collection_face.bin'  # Đường dẫn đến index 2

# Đọc index 1 và index 2 từ file
index1 = faiss.read_index(index_path_1)
index2 = faiss.read_index(index_path_2)
print("index1", index1.ntotal)
print("index2", index2.ntotal)

# Gộp hai indexes
merged_index = merge_indexes(index1, index2)

print("merged_index", merged_index.ntotal)

# # Lưu merged_index thành file mới
# merged_index_path = 'merged_index.index'
# faiss.write_index(merged_index, merged_index_path)