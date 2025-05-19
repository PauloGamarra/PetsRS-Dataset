import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
from sklearn.preprocessing import normalize

VERSION = 'paper'

def evaluate_knn(encodings_path, k):
    query_dir = os.path.join(encodings_path, 'query')
    candidate_dir = os.path.join(encodings_path , 'candidate')

    query_files = os.listdir(query_dir)
    candidate_files = os.listdir(candidate_dir)

    query_files.sort(key=lambda x: int(x.split('.')[0]))
    candidate_files = query_files.copy()

    query_path = os.path.join(query_dir, query_files[0])
    candidate_path = os.path.join(candidate_dir, candidate_files[0])

    query_encodings = np.load(query_path)
    candidate_encodings = np.load(candidate_path)

    for query_file in query_files[1:]:
        query_path = os.path.join(query_dir, query_file)
        candidate_path = os.path.join(candidate_dir, query_file)
        query_encodings = np.concatenate([query_encodings, np.load(query_path)])
        candidate_encodings = np.concatenate([candidate_encodings, np.load(candidate_path)])


    
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    
    # norm_candidate_encodings = normalize(candidate_encodings, norm='l2')
    # knn.fit(norm_candidate_encodings)

    knn.fit(candidate_encodings)
    
    hits = 0 
    for idx in range(len(query_encodings)):
        
        # norm_query_encodings = normalize(query_encodings[idx].reshape(1,-1), norm='l2')
        # _, neighbors_idx = knn.kneighbors(norm_query_encodings)
        
        _, neighbors_idx = knn.kneighbors(query_encodings[idx].reshape(1,-1))
        
        query_results = [candidate_files[cand_idx] for cand_idx in neighbors_idx[0]]
        hit = query_files[idx] in query_results
        if hit:
            hits += 1
    acc = hits / len(query_encodings)
    print('acc: {}'.format(round(acc*100,2)))

    return


def evaluate_encodings(encodings_dir, k):
    if VERSION == 'tcc':
        pre_processings = ['crop_original', 'crop_segmented']
        origins = ['desaparecido', 'procurase_dono']
        speciess = ['cao', 'gato']
        for pre_processing in pre_processings:
            for origin in origins:
                for species in speciess:
                    results_path = os.path.join(os.path.join(encodings_dir, pre_processing),os.path.join(origin, species))
                    print(results_path)
                    evaluate_knn(results_path, k)
    if VERSION == 'paper':
        origins = ['lost', 'found']
        speciess = ['dog', 'cat']
        for origin in origins:
            for species in speciess:
                results_path = os.path.join(encodings_dir, os.path.join(origin, species))
                print(results_path)
                evaluate_knn(results_path, k)



if __name__ == '__main__':
    results_dir = sys.argv[1]
    k = int(sys.argv[2])
    print('evaluating encodings on {} with k = {}'.format(results_dir, k))
    evaluate_encodings(results_dir, k)
