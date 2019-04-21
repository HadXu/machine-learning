# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       haxu
   date：          2019/4/14
-------------------------------------------------
   Change Activity:
                   2019/4/14:
-------------------------------------------------
"""
__author__ = 'haxu'

from argparse import Namespace
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim


class Vocabulary(object):
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class NMTVectorizer(object):
    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        """
        Args:
            source_vocab (SequenceVocabulary): maps source words to integers
            target_vocab (SequenceVocabulary): maps target words to integers
            max_source_length (int): the longest sequence in the source dataset
            max_target_length (int): the longest sequence in the target dataset
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """Vectorize the provided indices

        Args:
            indices (list): a list of integers that represent a sequence
            vector_length (int): an argument for forcing the length of index vector
            mask_index (int): the mask_index to use; almost always 0
        """
        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector

    def _get_source_indices(self, text):
        """Return the vectorized source text

        Args:
            text (str): the source text; tokens should be separated by spaces
        Returns:
            indices (list): list of integers representing the text
        """
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(self.source_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.source_vocab.end_seq_index)
        return indices

    def _get_target_indices(self, text):
        """Return the vectorized source text

        Args:
            text (str): the source text; tokens should be separated by spaces
        Returns:
            a tuple: (x_indices, y_indices)
                x_indices (list): list of integers representing the observations in target decoder
                y_indices (list): list of integers representing predictions in target decoder
        """
        indices = [self.target_vocab.lookup_token(token) for token in text.split(" ")]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]
        return x_indices, y_indices

    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        source_vector_length = -1
        target_vector_length = -1

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2  # begin end
            target_vector_length = self.max_target_length + 1  # end

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices,
                                        vector_length=source_vector_length,
                                        mask_index=self.source_vocab.mask_index)

        target_x_indices, target_y_indices = self._get_target_indices(target_text)

        target_x_vector = self._vectorize(target_x_indices,
                                          vector_length=target_vector_length,
                                          mask_index=self.target_vocab.mask_index)
        target_y_vector = self._vectorize(target_y_indices,
                                          vector_length=target_vector_length,
                                          mask_index=self.target_vocab.mask_index)

        return {"source_vector": source_vector,
                "target_x_vector": target_x_vector,
                "target_y_vector": target_y_vector,
                "source_length": len(source_indices)}

    @classmethod
    def from_dataframe(cls, bitext_df):
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()

        max_source_length = 0
        max_target_length = 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])

        return cls(source_vocab=source_vocab,
                   target_vocab=target_vocab,
                   max_source_length=contents["max_source_length"],
                   max_target_length=contents["max_target_length"])

    def to_serializable(self):
        return {"source_vocab": self.source_vocab.to_serializable(),
                "target_vocab": self.target_vocab.to_serializable(),
                "max_source_length": self.max_source_length,
                "max_target_length": self.max_target_length}


class NMTDataset(Dataset):
    def __init__(self, text_df, vectorizer):
        self.text_df = text_df
        self._vectorizer = vectorizer

        self.train_df = self.text_df[self.text_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        text_df = pd.read_csv(dataset_csv)
        train_subset = text_df[text_df.split == 'train']
        return cls(text_df, NMTVectorizer.from_dataframe(train_subset))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_csv, vectorizer_filepath):
        text_df = pd.read_csv(dataset_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(text_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return NMTVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(row.source_language, row.target_language)

        return {"x_source": vector_dict["source_vector"],
                "x_target": vector_dict["target_x_vector"],
                "y_target": vector_dict["target_y_vector"],
                "x_source_length": vector_dict["source_length"]}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def generate_nmt_batches(dataset, batch_size, shuffle=False,
                         drop_last=True, device="cpu"):
    """A generator function which wraps the PyTorch DataLoader.  The NMT Version """
    """ 同时对长度进行排序 从大到小"""
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        lengths = data_dict['x_source_length'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict


class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        super(NMTEncoder, self).__init__()
        self.source_embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_size,
            padding_idx=0,
        )
        self.birnn = nn.GRU(
            embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True
        )

    def forward(self, x_source, x_lengths):
        """
        :param x_source: （bs, 25）
        :param x_lengths: (bs, )
        :return:
        """
        x_embeded = self.source_embedding(x_source)  # (bs, 25, 64)
        x_lengths = x_lengths.numpy()  # (bs,)

        x_packed = pack_padded_sequence(x_embeded, x_lengths, batch_first=True)  # (sum(x_lengths), 64)

        x_birnn_out, x_birnn_h = self.birnn(x_packed)  # [(sum(x_lengths), 128*2), (2, bs, 128)]

        x_birnn_h = x_birnn_h.permute(1, 0, 2)  # (bs, 2, 128)

        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)  # (bs, 256)

        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)  # (bs, ?,256)

        # (bs, 10, 256)
        # (bs, 256)
        return x_unpacked, x_birnn_h


def verbose_attention(encoder_state_vectors, query_vector):
    # (bs, max_len, 256)
    # (bs, 256)

    batch_size, num_vectors, vector_size = encoder_state_vectors.size()

    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size),
                              dim=2)  # (bs, max_len)

    vector_probabilities = F.softmax(vector_scores, dim=1)  # (bs, max_len)

    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size,
                                                                         num_vectors, 1)  # (bs, max_len, 256)

    context_vectors = torch.sum(weighted_vectors, dim=1)  # (bs, 256)

    return context_vectors, vector_probabilities, vector_scores


class NMTDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_index):
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_size,
                                             padding_idx=0)
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size,
                                   rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings)
        self.bos_index = bos_index
        self._sampling_temperature = 3

    def _init_indices(self, batch_size):
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index

    def _init_context_vectors(self, batch_size):
        return torch.zeros(batch_size, self._rnn_hidden_size)

    def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability=0.0):
        """

        :param encoder_state:  (bs, max_len, 256)
        :param initial_hidden_state: (bs, 256)
        :param target_sequence:  (bs, 25) target
        :param sample_probability:
        :return:
        """
        if target_sequence is None:
            sample_probability = 1.
        else:
            target_sequence = target_sequence.permute(1, 0)  # （25,bs）

        h_t = self.hidden_map(initial_hidden_state)  # (bs, 256)

        batch_size = encoder_state.size(0)  # bs

        context_vectors = self._init_context_vectors(batch_size)  # (bs, 256)

        y_t_index = self._init_indices(batch_size)  # (bs, ) [2] * bs

        device = encoder_state.device
        h_t = h_t.to(device)
        y_t_index = y_t_index.to(device)
        context_vectors = context_vectors.to(device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()  # (bs ,10, 256)

        output_sequence_size = target_sequence.size(0)  # 25

        for i in range(output_sequence_size):
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = target_sequence[i]

            y_input_vector = self.target_embedding(y_t_index)  # (bs, 64)

            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)  # (bs, 64 + 256)

            h_t = self.gru_cell(rnn_input, h_t)  # (bs, 256)

            self._cached_ht.append(h_t.cpu().data.numpy())

            # (bs, max_len, 256)
            # (bs, 256)

            # 输出
            # (bs ,256)
            # (bs, max_len)
            context_vectors, p_attn, _ = verbose_attention(
                encoder_state_vectors=encoder_state,
                query_vector=h_t,
            )

            self._cached_p_attn.append(p_attn.cpu().detach().numpy())

            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3))  # (bs, 4911)

            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()

            output_vectors.append(score_for_y_t_index)

        # (25, 5, 4911)
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)  # (bs, 25, 4911)

        return output_vectors


class NMTModel(nn.Module):
    def __init__(self, source_vocab_size, source_embedding_size,
                 target_vocab_size, target_embedding_size, encoding_size,
                 target_bos_index):
        super(NMTModel, self).__init__()
        self.encoder = NMTEncoder(num_embeddings=source_vocab_size,
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoder(num_embeddings=target_vocab_size,
                                  embedding_size=target_embedding_size,
                                  rnn_hidden_size=decoding_size,
                                  bos_index=target_bos_index)

    def forward(self, x_source, x_source_lengths, target_sequence, sample_probability=0.5):
        """
        :param x_source: (batch, vectorizer.max_source_length) (bs,25)
        :param x_source_lengths: length of the sequence (bs,)
        :param target_sequence: target text data tensor (bs, 25)
        :return: prediction vectors at each output step
        """
        # (bs, 10, 256)
        # (bs, 256)
        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)

        decoded_states = self.decoder(encoder_state,
                                      final_hidden_states,
                                      target_sequence,
                                      sample_probability=sample_probability,
                                      )

        return decoded_states


def normalize_sizes(y_pred, y_true):
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


if __name__ == '__main__':
    args = Namespace(
        dataset_csv="simplest_eng_fra.csv",
        vectorizer_file="vectorizer.json",
        learning_rate=5e-4,
        batch_size=5,
        source_embedding_size=64,
        target_embedding_size=64,
        encoding_size=128,
        device='cpu',
    )
    dataset = NMTDataset.load_dataset_and_make_vectorizer(args.dataset_csv)
    dataset.save_vectorizer(args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()
    mask_index = vectorizer.target_vocab.mask_index

    dataset.set_split('train')
    batch_generator = generate_nmt_batches(dataset,
                                           batch_size=args.batch_size,
                                           device=args.device)

    model = NMTModel(
        source_vocab_size=len(vectorizer.source_vocab),
        source_embedding_size=args.source_embedding_size,
        target_vocab_size=len(vectorizer.target_vocab),
        target_embedding_size=args.target_embedding_size,
        encoding_size=args.encoding_size,
        target_bos_index=vectorizer.target_vocab.begin_seq_index
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for batch_idx, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()

        y_pred = model(batch_dict['x_source'],
                       batch_dict['x_source_length'],
                       batch_dict['x_target'],
                       sample_probability=0.5,
                       )

        loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)
        loss.backward()
        optimizer.step()

        print(loss.item())

