import torch
from transformers import AutoTokenizer, EsmModel, BatchEncoding
from typing import List, Dict
import os


class ESM2Encoder(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "../esm2_model"
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Pre-trained model to load.
        """
        super(ESM2Encoder, self).__init__()
        assert pretrained_model_name_or_path is not None
        
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        model_path = os.path.join(repo_root, 'esm2_model')

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = EsmModel.from_pretrained(model_path, local_files_only=True)

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocabs(self) -> List[str]:
        return self.tokenizer.all_tokens

    @property
    def id2token(self) -> Dict[int, str]:
        return self.tokenzier._id_to_token

    @property
    def token2id(self) -> Dict[str, int]:
        return self.tokenizer._token_to_id

    def tokenize(self, inputs: List[str]) -> BatchEncoding:
        """Convert inputs to a format suitable for the model.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [population].

        Returns:
            encoded_inputs (BatchEncoding): a BatchEncoding object.
        """
        encoded_inputs = self.tokenizer(inputs,
                                        add_special_tokens=True,
                                        return_tensors="pt",
                                        padding=True)
        return encoded_inputs

    def decode(self, tokens: torch.Tensor) -> List[str]:
        """Decode predicted tokens into alphabet characters

        Args:
            tokens (torch.Tensor): Predicted tokens of shape [batch, sequence_length]

        Returns:
            (List[str]): Predicted characters.
        """
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        """Forward pass of ESM2 model

        Args:
            inputs (BatchEncoding): Output of tokenizer.

        Returns:
            state (torch.Tensor): Hidden representation of shape `[batch, seq_len, dim]`.
        """
        state = self.model(**inputs).last_hidden_state  # [B, L, D]
        return state
