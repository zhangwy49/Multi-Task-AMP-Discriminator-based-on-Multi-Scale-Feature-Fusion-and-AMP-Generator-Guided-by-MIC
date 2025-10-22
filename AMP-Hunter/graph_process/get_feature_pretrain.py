import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, EsmModel

print('start')

def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sequences = [line.split()[1] for line in lines]  
    return sequences


def get_sequence_embeddings(sequences):
    embeddings = []
    for seq in sequences:
        inputs = tokenizer(seq, padding=True, return_tensors="pt", truncation = True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
   
        last_hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
        
        # use average pooling to get the sequence embedding
        avg_embedding = last_hidden_states.mean(dim=1)  # Shape: [batch_size, hidden_dim]
        
        embeddings.append(avg_embedding.cpu().numpy().squeeze())

    return np.array(embeddings)


def process_class_file(input_file, output_file, positive_count=3280):
    line_index = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            content = ",".join(line.split())

            label = 1 if i < positive_count else 0

            new_line = f"{line_index}\t{content}\t{label}\n"
            fout.write(new_line)
            line_index += 1

def process_reg_file(input_file, data_file, output_file):
    line_index = 0

    labels = []
    with open(data_file, 'r', encoding='utf-8') as fdata:
        for line in fdata:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.append(parts[2])  

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            content = ",".join(line.split())

            label = labels[i] if i < len(labels) else "0"

            new_line = f"{line_index}\t{content}\t{label}\n"
            fout.write(new_line)
            line_index += 1




if __name__ == "__main__":
    #loading model
    ems2_model_path = '../../AMP-Forge/esm2_model'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ems2_model_path)
    model = EsmModel.from_pretrained(ems2_model_path).to(device)
    model.eval()  # disables dropout for deterministic results

    #get sequence embedding
    sequences = read_data('../AMP_dataset/raw/data.txt')
    embeddings = get_sequence_embeddings(sequences)
    np.savetxt('../AMP_dataset/processed/raw_embeddings.txt', embeddings)
    data_file = '../AMP_dataset/raw/data.txt'
    input_file = "../AMP_dataset/processed/raw_embeddings.txt"
    output_class_file = "node_feature_label_class.txt"
    output_reg_file = "node_feature_label_reg.txt"

    process_class_file(input_file, output_class_file, positive_count=3280)
    process_reg_file(input_file, data_file, output_reg_file)

