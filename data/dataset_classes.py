import torch
import lmdb
import pickle
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset
from typing import List, Dict, Optional


class SequenceDatasetFromList(TorchDataset):
    def __init__(self, sequences, **kwargs):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class LMDBFeatureStore:
    """Lightweight wrapper around an LMDB env storing pickled feature dicts."""

    def __init__(self, path: str):
        self.env = lmdb.open(path, readonly=True, lock=False)

    def get(self, key: str) -> Optional[Dict[str, List]]:
        with self.env.begin() as txn:
            value = txn.get(key.encode())
        if value is None:
            return None
        return pickle.loads(value)


class LMDBSequenceDataset(IterableDataset):
    """Read token sequences and optional features from LMDB."""

    def __init__(self, seq_lmdb_path: str, feature_store: Optional[LMDBFeatureStore] = None):
        self.seq_env = lmdb.open(seq_lmdb_path, readonly=True, lock=False)
        self.feature_store = feature_store

    def __iter__(self):
        with self.seq_env.begin() as txn:
            length_bytes = txn.get(b'length')
            length = int(length_bytes) if length_bytes else 0
            for i in range(length):
                key = str(i)
                tokens = txn.get(key.encode())
                if tokens is None:
                    continue
                token_ids = pickle.loads(tokens)
                item = {'tokens': token_ids}
                if self.feature_store:
                    feats = self.feature_store.get(key) or {}
                    item.update(feats)
                yield item


class IterableDatasetFromHF(IterableDataset):
    def __init__(self, dataset, col_name='seqs', **kwargs):
        """
        Wrap a streaming Hugging Face dataset (IterableDataset) into a PyTorch IterableDataset.
        
        Args:
            dataset (IterableDataset): Streaming Hugging Face dataset.
            col_name (str): The column name containing the sequences.
        """
        self.dataset = dataset
        self.col_name = col_name

    def __iter__(self):
        for example in self.dataset:
            yield example[self.col_name]


class SequenceCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.eos_token = tokenizer.eos_token

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        seq = ''.join([self.cls_token + s + self.eos_token for s in batch])
        input_ids = self.tokenizer.encode(seq, add_special_tokens=False, return_tensors='pt')
        return {'input_ids':input_ids}


class TokenBasedIterableDataset(IterableDataset):
    def __init__(self, dataset, target_token_count=8192, col_name='seqs',
                 numeric_feature_keys=None, token_feature_keys=None,
                 feature_store: Optional[LMDBFeatureStore] = None,
                 tokens_key: str = 'tokens', id_key: str = 'id', **kwargs):
        """
        Wrap a streaming dataset to yield batches based on token count rather than sequence count.
        
        Args:
            dataset (IterableDataset): Streaming Hugging Face dataset
            tokenizer: Tokenizer to use for counting tokens
            target_token_count (int): Target number of tokens per batch
            col_name (str): Column name containing sequences
        """
        self.dataset = dataset
        self.target_token_count = target_token_count
        self.col_name = col_name
        self.numeric_feature_keys = numeric_feature_keys or []
        self.token_feature_keys = token_feature_keys or []
        self.feature_store = feature_store
        self.tokens_key = tokens_key
        self.id_key = id_key

    def __iter__(self):
        accumulated_sequences = []
        accumulated_numeric = []
        accumulated_tokens = []
        current_token_count = 0
        for idx, example in enumerate(self.dataset):
            if self.tokens_key in example:
                seq = example[self.tokens_key]
            else:
                seq = example[self.col_name]
            seq_token_count = len(seq) + 2  # +2 for cls and eos tokens

            if self.feature_store:
                key = str(example.get(self.id_key, idx))
                feats = self.feature_store.get(key) or {}
                numeric_feats = feats.get('numeric_features', [])
                token_feats = feats.get('token_features', [])
            else:
                numeric_feats = [example.get(k, 0.0) for k in self.numeric_feature_keys]
                token_feats = [example.get(k, 0) for k in self.token_feature_keys]
            
            # If adding this sequence would exceed target and we have accumulated sequences, yield batch
            if current_token_count + seq_token_count > self.target_token_count and accumulated_sequences:
                yield {
                    'sequences': accumulated_sequences,
                    'numeric_features': accumulated_numeric,
                    'token_features': accumulated_tokens,
                }
                accumulated_sequences = []
                accumulated_numeric = []
                accumulated_tokens = []
                current_token_count = 0
            
            accumulated_sequences.append(seq)
            accumulated_numeric.append(numeric_feats)
            accumulated_tokens.append(token_feats)
            current_token_count += seq_token_count
            
            # If we've reached the target, yield batch
            if current_token_count >= self.target_token_count:
                yield {
                    'sequences': accumulated_sequences,
                    'numeric_features': accumulated_numeric,
                    'token_features': accumulated_tokens,
                }
                accumulated_sequences = []
                accumulated_numeric = []
                accumulated_tokens = []
                current_token_count = 0
        
        # Yield any remaining sequences
        if accumulated_sequences:
            yield {
                'sequences': accumulated_sequences,
                'numeric_features': accumulated_numeric,
                'token_features': accumulated_tokens,
            }


class TokenBasedSequenceCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.eos_token = tokenizer.eos_token
        self.cls_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        item = batch[0]
        if isinstance(item, list):
            sequences = item
            numeric_feats = []
            token_feats = []
        elif isinstance(item, dict):
            sequences = item.get('sequences', [])
            numeric_feats = item.get('numeric_features', [])
            token_feats = item.get('token_features', [])
        else:
            sequences = [item]
            numeric_feats = []
            token_feats = []

        if sequences and isinstance(sequences[0], (list, tuple)):
            tokens = []
            for seq in sequences:
                tokens += [self.cls_id] + list(seq) + [self.eos_id]
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        else:
            seq = ''.join([self.cls_token + s + self.eos_token for s in sequences])
            input_ids = self.tokenizer.encode(seq, add_special_tokens=False, return_tensors='pt')
        batch_dict = {'input_ids': input_ids}
        if numeric_feats:
            batch_dict['numeric_features'] = torch.tensor(numeric_feats, dtype=torch.float32)
        if token_feats:
            batch_dict['token_features'] = torch.tensor(token_feats, dtype=torch.long)
        return batch_dict

