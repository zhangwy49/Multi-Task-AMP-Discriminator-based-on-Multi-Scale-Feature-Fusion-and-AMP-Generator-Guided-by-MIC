# Multi Task AMP Discriminator based on Multi Scale Feature Fusion and AMP Generator Guided by MIC
### This framework includes 2 parts:  
**AMP-Hunter** (a multi-task discriminator that fuses convolutional neural networks with graph neural networks) in AMP classification and MIC predictiontask.  
**AMP-Forge** (a generator that integrates multi sequence alignment method to select original candidates) and is guided by minimum inhibitory concentration (MIC) to optimize sequences.  

![framework](framework.jpg)  

# Python Environment Setup
For **AMP-Hunter**，the required runtime environment configuration file is: [environment4discrimination.yaml](./AMP-Hunter/environment4discrimination.yaml)  

The relevant environment can be configured using the following command:
```
conda env create -f environment4discrimination.yaml
source activate AMP-Hunter
conda activate AMP-Hunter
```

For **AMP-Forge**，the required runtime environment configuration file is: [environment4generation.yaml](./AMP-Forge/environment4generation.yaml)  

The relevant environment can be configured using the following command:
```
conda env create -f environment4generation.yaml
source activate AMP-Forge
conda activate AMP-Forge
```

# Download ESM2 model  
The ESM2 model is used in AMP-Hunter to extract features as node features, and serve as an encoder for extract residual-level features of the sequences in AMP-Forge.  
Please download the required files.  
👉 Recommended download (config.json, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, vocab.txt) from the [huggingface](https://huggingface.co/Rocketknight1/esm2_t33_650M_UR50D/tree/main) 
and place them in the [./AMP-Forge/esm2_model](./AMP-Forge/esm2_model) path.  

# Preprocess Data for AMP-Hunter  
### obtain the sequence-level features extracted by esm2  
In the [./AMP-Forge/esm2_model](./AMP-Forge/esm2_model) path


