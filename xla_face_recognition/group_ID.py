import faiss
import cv2
import numpy as np
import sys

def similarity_scores_features(features):
    concatenated_features = np.concatenate(features, axis=0)
    transposed_features = concatenated_features.T
    similarity_scores = np.matmul(concatenated_features, transposed_features)
    return similarity_scores

def group_id(list_ID_group: list, src_index, src_names, src_index_collection, src_names_collection, threshold_take = 0.35):
    length_src_or_collection = []
    ind_embeddings_process_ID = []
    dem = 0
    embeddings_process_ID = []
    for id in list_ID_group:
        if "ID" in id:
            indices_ID = [idx for idx, name in enumerate(src_names_collection) if name == id]
            if not indices_ID:
                length_src_or_collection.append(0)
                ind_embeddings_process_ID.append(0)
            else:
                dem += len(indices_ID)
                embeddings_process_ID += [np.array(src_index_collection.reconstruct(indices_ID[0])).reshape((1,512))]
                length_src_or_collection.append(len(indices_ID))
                ind_embeddings_process_ID.append(dem)
                # print("====")
                # print("length_src_or_collection: ", length_src_or_collection)
                # print("ind_embeddings_process_ID: ", ind_embeddings_process_ID)

        else:
            indices_ID = [idx for idx, name in enumerate(src_names) if name == id]
            if not indices_ID:
                length_src_or_collection.append(0)
                ind_embeddings_process_ID.append(0)
            else:
                dem += len(indices_ID)
                embeddings_process_ID += [np.array(src_index.reconstruct(indices_ID[0])).reshape((1,512))]
                length_src_or_collection.append(len(indices_ID))
                ind_embeddings_process_ID.append(dem)

    lst_accept_element = []
    if not embeddings_process_ID:
        pass

    else:
        similarity_scores = similarity_scores_features(embeddings_process_ID)
        print("similarity_scores: ", similarity_scores)

        # chia ra lam 3 khoang tinh do tuong dong 0.35 -> 100% gom 3 khoang: < 0.1, 0.1 -> <0.2, > 0.2
        array_row_head = similarity_scores[0]
        case_1 = []
        case_2 = []
        case_3 = []
        for i in range(len(array_row_head)):
            if i != 0:
                if array_row_head[i] < 0.1:
                    case_1.append(array_row_head[i])

                elif array_row_head[i] > 0.2:
                    case_3.append(array_row_head[i])

                else:
                    case_2.append(array_row_head[i])
    

    # print("case 1: ", case_1)
    # print("case 2: ", case_2)
    # print("case 3: ", case_3)


        # for i in range(len(length_src_or_collection)):
        #     end_col = ind_embeddings_process_ID[i] - 1
        #     start_col = ind_embeddings_process_ID[i] - length_src_or_collection[i]
        #     start_row = 0
        #     has_same = False
        #     for j in range(len(length_src_or_collection)):
        #         end_row = start_row + length_src_or_collection[j] - 1
        #         if i != j:
        #             sum_range = np.sum(similarity_scores[start_row:end_row+1, start_col:end_col+1])
        #             # print("++++")
        #             # print(similarity_scores[start_row:end_row+1, start_col:end_col+1])

        #             # Chia tổng cho số phần tử trong phạm vi
        #             num_elements = (end_row - start_row + 1) * (end_col - start_col + 1)
        #             average_range = sum_range / num_elements
        #             if average_range >= threshold_take:
        #                 has_same = True
            
        #     lst_accept_element.append(has_same)
    
    return lst_accept_element

def creat_new_data(embedding_search, src_index, src_names, src_index_collection, src_names_collection, threshold_take = 0.3, num_search = 20):
    D_src, I_src = src_index.search(embedding_search, num_search)
    D_collection, I_collection = src_index_collection.search(embedding_search, num_search)

    dict_ID = {}

    for i in range(num_search):
        if D_collection[0][i] >= threshold_take:
            if src_names_collection[I_collection[0][i]] not in dict_ID.keys():
                dict_ID[src_names_collection[I_collection[0][i]]] = D_collection[0][i]
            elif dict_ID[src_names_collection[I_collection[0][i]]] < D_collection[0][i]:
                dict_ID[src_names_collection[I_collection[0][i]]] = D_collection[0][i]
        
        if D_src[0][i] >= threshold_take:
            if src_names[I_src[0][i]] not in dict_ID.keys():
                dict_ID[src_names[I_src[0][i]]] = D_src[0][i]
            elif dict_ID[src_names[I_src[0][i]]] < D_src[0][i]:
                dict_ID[src_names[I_src[0][i]]] = D_src[0][i]

    print('D_collection: ',D_collection)
    # print(I_collection[0][0])
    # print(I_collection[0][1])
    # print(I_collection[0][2])
    # print(I_collection[0][3])
    # print(I_collection[0][4])
    print(dict_ID)

def merge_ID(list_id : list, src_names, src_names_collection):
    fname = list_id[0]
    for id in list_id:
        if "ID" in id:
            for i in range(len(src_names_collection)):
                if src_names_collection[i] == id:
                    src_names_collection[i] = fname
        else:
           for i in range(len(src_names)):
                if src_names[i] == id:
                    src_names[i] = fname



# test
if __name__ == '__main__':
    src_embedded = faiss.read_index("/home/maicg/Documents/Me/python-image-processing/xla_face_recognition/data/cerberus_ada_r50.bin")
    src_names = np.load("/home/maicg/Documents/Me/python-image-processing/xla_face_recognition/data/cerberus_ada_r50.npy", allow_pickle=True).tolist()

    src_index_collection = faiss.read_index("/home/maicg/Documents/Me/CERBERUS/GitLab/cccdserverdjango/server.bin")
    src_names_collection = np.load("/home/maicg/Documents/Me/CERBERUS/GitLab/cccdserverdjango/server.npy", allow_pickle=True).tolist()

    # src_index_collection = faiss.read_index("/home/maicg/Documents/Me/python-image-processing/xla_face_recognition/data/collection_face.bin")
    # src_names_collection = np.load("/home/maicg/Documents/Me/python-image-processing/xla_face_recognition/data/collection_face.npy", allow_pickle=True).tolist()

    # src_index_collection = faiss.read_index("/home/maicg/Documents/Me/python-image-processing/xla_face_recognition/data/faiss_07/collection_face.bin")
    # src_names_collection =  np.load("/home/maicg/Documents/Me/python-image-processing/xla_face_recognition/data/faiss_07/collection_face.npy", allow_pickle=True).tolist()

    src_search = faiss.read_index("/home/maicg/Documents/Me/python-image-processing/xla_face_recognition/data/faiss_07/collection_face.bin")
    src_name_search =  np.load("/home/maicg/Documents/Me/python-image-processing/xla_face_recognition/data/faiss_07/collection_face.npy", allow_pickle=True).tolist()


    print("src_embedded", src_embedded.ntotal)
    print("src_names", len(src_names))

    print("src_index_collection", src_index_collection.ntotal)
    print("src_names_collection", len(src_names_collection))

    list_ID = ['ID4', 'ID5', 'ID11', 'ID511']
    # list_ID = ['ID0', 'ID2']
    # list_ID = ['ID6', 'ID25']
    # list_ID = ['ID7', 'ID24']
    # list_ID = ['ID587', 'ID586', 'ID522', 'ID521', 'ID520', 'ID519']

    results = group_id(list_ID, src_embedded, src_names, src_index_collection, src_names_collection)

    print(results)
    # image = sys.argv[1]
    # image = cv2.imread(image)
    id = sys.argv[1]
    ind_name_search = [idx for idx, name in enumerate(src_name_search) if name == id]
    embedding = np.array(src_search.reconstruct(ind_name_search[0])).reshape((1,512))
    creat_new_data(embedding, src_embedded, src_names, src_index_collection, src_names_collection)

    # print("===")
    # print(src_name_search)
    # print("===")
    # for i in range(len(src_name_search)):
    #     if src_name_search[i] == 'ID1067':
    #         src_name_search[i] = "id"
    
    # print(src_name_search)            

                        

                
                
