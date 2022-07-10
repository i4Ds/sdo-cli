"""
Inspired by https://gist.github.com/wassname/8ae1f64389c2aaceeb84fcd34c3651c3

A Pytorch sampler that samples ordered indices from unordered sequences. 
Good for the use with Dask because Dask will slow down when sampling between chunks.
Usually, it is better if batches are uncorrelated so we want each batch to be sequence from a different part of a chunk.
For example, given each chunk is `range(12)`. Our seq_len is 3. We might end up with these indices:
- [[1,2,3],[9,10,11],[4,5,6]]
Usage:
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SequenceInChunkSampler(train_dataset, seq_len=batch_size, chunksize=batch_size*100)
    )
"""
import torch.utils.data.sampler
import numpy as np


class SequenceInChunkSampler(torch.utils.data.sampler.Sampler):
    """
    Samples sequences of elements sequentially, but random sequences in a chunk.
    Arguments:
        data_source (Dataset): dataset to sample from
        seq_len (int): length of sequential sequences
        chunksize (int): length of cached data to take random sequences from
    """

    def __init__(self, data_source, seq_len=12, chunksize=120):
        assert chunksize % seq_len == 0, "chunk size should be a multiple of seq_len"
        assert len(data_source) > chunksize
        self.data_source = data_source
        self.seq_len = seq_len
        self.chunksize = chunksize

    def __iter__(self):
        chunk_idxs = np.arange(0, len(self.data_source), self.chunksize)
        max_i = len(self.data_source)
        np.random.shuffle(chunk_idxs)
        for chunk_idx in chunk_idxs:
            seqs = np.arange(
                chunk_idx, min(chunk_idx + self.chunksize, max_i), self.seq_len
            )
            np.random.shuffle(seqs)
            for seq_i in seqs:
                for i in np.arange(seq_i, min(seq_i + self.seq_len, max_i)):
                    yield i

    def __len__(self):
        return len(self.data_source)
