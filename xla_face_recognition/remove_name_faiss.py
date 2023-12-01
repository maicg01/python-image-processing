import cv2
import numpy as np
import faiss

# tinh similarity cua nhieu embedding va tra ve mot matric
# dau vao la mot mang numpy, neu la tensor thi dung
# similarity_scores = torch.cat(features) @ torch.cat(features).T
def similarity_scores_features(features):
    concatenated_features = np.concatenate(features, axis=0)
    transposed_features = concatenated_features.T
    similarity_scores = np.matmul(concatenated_features, transposed_features)
    return similarity_scores

def indices_embedding_error(src_index, src_names, name_ID, threshold_take_embed_process = 0.4, threshold_take_embed_remove = 0.45):
    indices_ID = [idx for idx, name in enumerate(src_names) if name == name_ID]
    if not indices_ID:
        print("No vectors found with the target name is {}.".format(name_ID))
        return []
    else:
        embeddings_process_ID = [np.array(src_index.reconstruct(idx)).reshape((1,512)) for idx in indices_ID]
        similarity_scores = similarity_scores_features(embeddings_process_ID)
        print("============: ", similarity_scores)
        array_row_head = similarity_scores[0]
        indices_remove = []
        for index_embed in range(len(indices_ID)):
            if array_row_head[index_embed] <= threshold_take_embed_process:
                average_row = (np.sum(similarity_scores[index_embed]) - 1)/(len(indices_ID) - 1)
                if average_row <= threshold_take_embed_remove:
                    indices_remove.append(indices_ID[index_embed])
        if not indices_remove:
            print("No vectors is removed")
            return []
        return indices_remove


def remove_embedding_with_indices(src_index, src_names, lst_indeces_to_remove):
    # Remove the vectors and names from the source index
    src_index.remove_ids(np.array(lst_indeces_to_remove))
    for idx in reversed(lst_indeces_to_remove):
        src_names.pop(idx)

# test
if __name__ == '__main__':
    src_embedded = faiss.read_index("collection_face2.bin")
    src_names = np.load("collection_face2.npy", allow_pickle=True).tolist()
    print("src_embedded", src_embedded.ntotal)

    list_remove = indices_embedding_error(src_embedded, src_names, name_ID="ID18")
    print(list_remove)

    if len(list_remove) > 0:
        remove_embedding_with_indices(src_embedded, src_names, list_remove)

    # # Save the modified indexes and names to their respective files
    # faiss.write_index(src_embedded, "collection_face2.bin")
    # np.save("collection_face2.npy", np.array(src_names), allow_pickle=True)
