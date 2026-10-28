"""Vocabulary manager."""
from typing import Dict, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    _shared_vocab: Dict[str, int] = {}
    _shared_id_to_token: Dict[int, str] = {}

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        self.pad_token_id: int = 0
        self.unk_token_id: int = 1
        self.bos_token_id: int = 50256
        self.eos_token_id: int = 50256

        self.add_token(PAD_TOKEN, 0)
        self.add_token(UNK_TOKEN, 1)
        self.add_token(BOS_TOKEN, 50256)
        self._token_to_id[EOS_TOKEN] = self.eos_token_id

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            if idx is not None and self._token_to_id[token] != idx:
                raise ValueError(f"Token {token!r} already has id {self._token_to_id[token]}")
            return self._token_to_id[token]
        if idx is None:
            idx = next((candidate for candidate in range(2, 50256) if candidate not in self._id_to_token),
                       max(self._id_to_token, default=-1) + 1)
        if idx < 0:
            raise ValueError("token id must be non-negative")
        if idx in self._id_to_token and self._id_to_token[idx] != token:
            raise ValueError(f"Token id {idx} is already assigned")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        return UNK_TOKEN

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
