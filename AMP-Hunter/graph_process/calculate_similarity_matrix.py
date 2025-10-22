import numpy as np
import parasail
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from math import sqrt

logging.basicConfig(filename='parasail_0_all.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def read_sequences(file_path):
    logging.info(f"Reading sequences from {file_path}")
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.split('\t')
            #read sequence column
            sequences.append(fields[1].strip())  
    logging.info(f"Read {len(sequences)} sequences")
    return sequences

# calculate the similarity by pam250 matrix
def calculate_similarity(seq1, seq2):
    result = parasail.sw_trace_striped_16(seq1, seq2, 0, 1, parasail.pam250) 
    return result.score

# calculate the similarity by pam250 matrix
def calculate_similarity_for_pairs(start, end, sequences):
    logging.info(f"Processing sequences from {start} to {end}")
    partial_matrix = np.zeros((end - start, len(sequences)))
    self_matrix = np.zeros((1, end - start))
    for i in range(start, end):
        similarity_self = parasail.sw_trace_striped_16(sequences[i], sequences[i], 11, 1, parasail.pam250).score 
        self_matrix[0, i - start] = similarity_self
        for j in range(i, len(sequences)):
            similarity = parasail.sw_trace_striped_16(sequences[i], sequences[j], 11, 1, parasail.pam250).score 
            partial_matrix[i - start, j] = similarity
        
    logging.info(f"Finished processing sequences from {start} to {end}")
    return partial_matrix, self_matrix, start, end


def calculate_similarity_matrix_parallel(sequences, num_workers=20):
    num_sequences = len(sequences)
    # make sure every worker process at least 1 sequence
    chunk_size = max(1, num_sequences // num_workers)  
    similarity_matrix = np.zeros((num_sequences, num_sequences))
    similarity_self_matrix = np.zeros((1, num_sequences))
    
    logging.info(f"Starting parallel computation with {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # every worker（i_start to i_end raw）
        for i_start in range(0, num_sequences, chunk_size):
            i_end = min(i_start + chunk_size, num_sequences)
            futures.append(
                executor.submit(
                    calculate_similarity_for_pairs,
                    i_start, i_end, sequences
                )
            )

        for future in as_completed(futures):
            partial_matrix, self_matrix, i_start, i_end = future.result()
            similarity_matrix[i_start:i_end, :] = partial_matrix
            similarity_self_matrix[0, i_start:i_end] = self_matrix


    # save the whole matrix
    full_filename = "../AMP_dataset/processed/similarity_matrix.csv"
    np.savetxt(full_filename, similarity_matrix, delimiter=',', fmt='%.4f')
    logging.info(f"Saved full matrix to {full_filename}")

    full_self_filename = "../AMP_dataset/processed/similarity_self_matrix.csv"
    np.savetxt(full_self_filename, similarity_self_matrix, delimiter=',', fmt='%.4f')
    logging.info(f"Saved full matrix to {full_filename}")

    return similarity_matrix, similarity_self_matrix

# calculate the similarity of pairs of sequence in parallel
def calculate_normalized_similarity_for_pairs(start, end, raw_matrix, selfcompare_list):
    logging.info(f"Processing sequences from {start} to {end}")
    num_sequences = len(raw_matrix)
    partial_matrix = np.zeros((end - start, num_sequences))
    for i in range(start, end):
        for j in range(i,num_sequences):
            self_value = sqrt(selfcompare_list[i] * selfcompare_list[j])
            nor_value = raw_matrix[i,j] / self_value
            partial_matrix[i - start, j] = nor_value

        
    logging.info(f"Finished processing sequences from {start} to {end}")
    return partial_matrix, start, end


def calculate_normalize_similarity_matrix_parallel(raw_matrix, selfcompare_list, num_workers=20):
    num_sequences = len(raw_matrix)
    chunk_size = max(1, num_sequences // num_workers)  
    nor_similarity_matrix = np.zeros((num_sequences, num_sequences))
    
    logging.info(f"Starting normalize parallel computation with {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for i_start in range(0, num_sequences, chunk_size):
            i_end = min(i_start + chunk_size, num_sequences)
            futures.append(
                executor.submit(
                    calculate_normalized_similarity_for_pairs,
                    i_start, i_end, raw_matrix, selfcompare_list
                )
            )


        for future in as_completed(futures):
            partial_matrix, i_start, i_end = future.result()
            nor_similarity_matrix[i_start:i_end, :] = partial_matrix
            

    full_filename = "../AMP_dataset/processed/normalize_similarity_matrix.csv"
    np.savetxt(full_filename, nor_similarity_matrix, delimiter=',', fmt='%.4f')
    logging.info(f"Saved full normalize matrix to {full_filename}")

    return nor_similarity_matrix

def main():
    logging.info("Program started")
    bio_sequences = read_sequences('../AMP_dataset/raw/data.txt') 
    print("Finishing reading raw sequence")
    similarity_matrix = calculate_similarity_matrix_parallel(bio_sequences, num_workers=20)  
    print("Finishing calculate similarity matrix and self_compare matrix")

    logging.info("Reading csv file to calculate self_normalize similarity matrix")
    raw_file_path = "../AMP_dataset/processed/similarity_matrix.csv"
    raw_matrix = np.loadtxt(raw_file_path, delimiter=',', dtype=float)
    selfcompare_file_path = "../AMP_dataset/processed/similarity_self_matrix.csv"
    selfcompare_list = np.loadtxt(selfcompare_file_path, delimiter=',', dtype=float)
    print("Finishing reading similarity matrix")
    similarity_matrix = calculate_normalize_similarity_matrix_parallel(raw_matrix, selfcompare_list, num_workers=20)  
    print("Finishing calculate normalize similarity matrix")

    logging.info("Calculate Similarity Matrix Program Finished")

    logging.info("Generate Edge Program started")

    raw_file_path = "../AMP_dataset/processed/normalize_similarity_matrix.csv"
    nor_matrix = np.loadtxt(raw_file_path, delimiter=',', dtype=float)

    indices = np.argwhere(
        (nor_matrix > 0.2) &  
        (np.triu(np.ones_like(nor_matrix, dtype=bool), k=1))  
    )

    pairs = [(i, j) for i, j in indices]

    output_file = "../AMP_dataset/processed/graph_edges.txt"
    with open(output_file, "w") as f:
        for i, j in pairs:
            f.write(f"{i}\t{j}\n")

    print(f"Finding {len(pairs)} pairs edges, have saved to {output_file}")


    logging.info("Generate Edge Program finished")

if __name__ == "__main__":
    main()
