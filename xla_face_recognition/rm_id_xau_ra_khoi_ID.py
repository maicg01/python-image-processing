import cv2
import numpy as np
import faiss
import sys

# tinh similarity cua nhieu embedding va tra ve mot matric
# dau vao la mot mang numpy, neu la tensor thi dung
# similarity_scores = torch.cat(features) @ torch.cat(features).T
# def similarity_scores_features(features):
#     concatenated_features = np.concatenate(features, axis=0)
#     transposed_features = concatenated_features.T
#     similarity_scores = np.matmul(concatenated_features, transposed_features)
#     return similarity_scores

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def indices_embedding_error(base_vecto, src_index, src_names, name_ID):
    indices_ID = [idx for idx, name in enumerate(src_names) if name == name_ID]
    if not indices_ID:
        print("No vectors found with the target name is {}.".format(name_ID))
        return []
    else:
        embeddings_process_ID = [np.array(src_index.reconstruct(idx)).reshape((1,512)) for idx in indices_ID]
        indices_remove = []
        for index_embed in range(len(indices_ID)):
            similarity_score = cosine_similarity(embeddings_process_ID[index_embed], base_vecto)
            if similarity_score > 0.47:
                indices_remove.append(indices_ID[index_embed])

        if not indices_remove:
            print("No vectors is removed")
            return []
        return indices_remove


def remove_embedding_with_indices(src_index, src_names, lst_indeces_to_remove):
    # Remove the vectors and names from the source index
    src_index.remove_ids(np.array(lst_indeces_to_remove))
    src_names = np.delete(src_names, lst_indeces_to_remove)

# test
if __name__ == '__main__':

    base_vecto = cv2.imread(sys.argv[1])

    src_embedded = faiss.read_index("collection_face2.bin")
    src_names = np.load("collection_face2.npy", allow_pickle=True).tolist()
    print("src_embedded", src_embedded.ntotal)

    list_remove = indices_embedding_error(base_vecto, src_embedded, src_names, name_ID="ID43")
    print(list_remove)

    if len(list_remove) > 0:
        remove_embedding_with_indices(src_embedded, src_names, list_remove)

    # # Save the modified indexes and names to their respective files
    # faiss.write_index(src_embedded, "collection_face2.bin")
    # np.save("collection_face2.npy", np.array(src_names), allow_pickle=True)
