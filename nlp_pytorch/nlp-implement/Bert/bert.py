import numpy as np
import torch
from torch import nn
from torch import optim
import re
from random import *
import math

maxlen = 30
batch_size = 6
max_pred = 5
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768*4
d_k = d_v = 64
n_segments = 2

text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)

sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
word_list = list(sorted(set(" ".join(sentences).split())))
word_dict = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
for i, w in enumerate(word_list):
	word_dict[w] = i + 4

number_dict = {i: w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)


token_list = []
for sentence in sentences:
	arr = [word_dict[s] for s in sentence.split()]
	token_list.append(arr)

def make_batch():
	batch = []
	positive = negative = 0
	while positive != batch_size/2 or negative != batch_size/2:
		tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences))
		tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index] # 随机选择两个句子

		input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
		segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

		n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15))))

		cand_maked_pos = [i for i, token in enumerate(input_ids)
							if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]

		shuffle(cand_maked_pos)

		# mask的位置以及mask掉的词
		masked_tokens, masked_pos = [], []
		for pos in cand_maked_pos[:n_pred]:
			masked_pos.append(pos)
			masked_tokens.append(input_ids[pos])
			if random() < 0.8:
				input_ids[pos] = word_dict['[MASK]']
			elif random() < 0.5:
				index = randint(0, vocab_size - 1)
				input_ids[pos] = word_dict[number_dict[index]]

		n_pad = maxlen - len(input_ids)

		input_ids.extend([0] * n_pad)
		segment_ids.extend([0] * n_pad)

		if max_pred > n_pred:
			n_pad = max_pred - n_pred
			masked_tokens.extend([0] * n_pad)
			masked_pos.extend([0] * n_pad)


		if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
			batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
			positive += 1

		elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
			batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
			negative += 1

	return batch



def get_attn_pad_mask(seq_q, seq_k):
	batch_size, len_q = seq_q.size()
	batch_size, len_k = seq_k.size()
	pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
	pad_attn_mask.expand(batch_size, len_q, len_k)
	return pad_attn_mask

def gelu(x):
	return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
	def __init__(self):
		super(Embedding, self).__init__()
		self.tok_embed = nn.Embedding(vocab_size, d_model)
		self.pos_embed = nn.Embedding(maxlen, d_model)
		self.seg_embed = nn.Embedding(n_segments, d_model)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, x, seg):
		seq_len = x.size(1)
		pos = torch.arange(seq_len, dtype=torch.long)
		pos = pos.unsqueeze(0).expand_as(x)
		embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
		return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
	def __init__(self):
		super(ScaledDotProductAttention, self).__init__()

	def forward(self, Q, K, V, attn_mask):
	 	scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
	 	scores.masked_fill_(attn_mask, -1e9)
	 	attn = nn.Softmax(dim=-1)(scores)
	 	context = torch.matmul(attn, V)
	 	return context, attn


class MultiHeadAttention(nn.Module):
	def __init__(self):
		super(MultiHeadAttention, self).__init__()
		self.W_Q = nn.Linear(d_model, d_k * n_heads)
		self.W_K = nn.Linear(d_model, d_k * n_heads)
		self.W_V = nn.Linear(d_model, d_v * n_heads)

	def forward(self, Q, K, V, attn_mask):
		residual, batch_size = Q, Q.size(0)
		q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
		k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
		v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
		attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
		context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
		context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
		output = nn.Linear(n_heads * d_v, d_model)(context)
		return nn.LayerNorm(d_model)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
	def __init__(self):
		super(PoswiseFeedForwardNet, self).__init__()
		self.fc1 = nn.Linear(d_model, d_ff)
		self.fc2 = nn.Linear(d_ff, d_model)

	def forward(self, x):
		return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
	def __init__(self):
		super(EncoderLayer, self).__init__()
		self.enc_self_attn = MultiHeadAttention()
		self.pos_ffn = PoswiseFeedForwardNet()

	def forward(self, enc_inputs, enc_self_attn_mask):
		enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
		enc_outputs = self.pos_ffn(enc_outputs)
		return enc_outputs, attn



class BERT(nn.Module):
	def __init__(self):
		super(BERT, self).__init__()
		self.embedding = Embedding()
		self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
		self.fc = nn.Linear(d_model, d_model)
		self.activ1 = nn.Tanh()
		self.linear = nn.Linear(d_model, d_model)
		self.activ2 = gelu
		self.norm = nn.LayerNorm(d_model)
		self.classifier = nn.Linear(d_model, 2)
		embed_weight = self.embedding.tok_embed.weight
		n_vocab, n_dim = embed_weight.size()
		self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
		self.decoder.weight = embed_weight
		self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))



	def forward(self, input_ids, segment_ids, masked_pos):
		output = self.embedding(input_ids, segment_ids)
		enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
		for layer in self.layers:
			output, enc_self_attn = layer(output, enc_self_attn_mask)		


		h_pooled = self.activ1(self.fc(output[:, 0]))
		logits_clsf = self.classifier(h_pooled)


		masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # 6 5 768
		h_masked = torch.gather(output, 1, masked_pos) # 将屏蔽掉的词的embedding取出来
		h_masked = self.norm(self.activ2(self.linear(h_masked)))
		logits_lm = self.decoder(h_masked) + self.decoder_bias

		return logits_lm, logits_clsf


if __name__ == '__main__':
	model = BERT()
	criterion = nn.CrossEntropyLoss()	
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	batch = make_batch()

	input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))

	for epoch in range(100):
		optimizer.zero_grad()
		logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
		loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
		loss_lm = (loss_lm.float()).mean()
		loss_clsf = criterion(logits_clsf, isNext)
		loss = loss_lm + loss_clsf # 上下文有没有预测对 以及 maks的词有没有预测对

		print(f'epoch:{epoch} -- loss:{loss.item()}')

		loss.backward()
		optimizer.step()


	input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
	print(text)
	print([number_dict[w] for w in input_ids if number_dict[w] != '[PAD]'])

	logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
	                           torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
	logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
	print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
	print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

	logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
	print('isNext : ', True if isNext else False)
	print('predict isNext : ',True if logits_clsf else False)




























































































