import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F

embedding_size = 2
sequence_length = 3
num_classes = 2
filter_sizes = [2, 2, 2]
num_filters = 3

sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]

labels = [1, 1, 1, 0, 0, 0]


word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)


inputs = []
for sen in sentences:
	inputs.append(np.asarray([word_dict[n] for n in sen.split()]))

targets = []
for out in labels:
	targets.append(out)

input_batch, target_batch = torch.LongTensor(inputs), torch.LongTensor(targets)


class TextCNN(nn.Module):
	def __init__(self):
		super(TextCNN, self).__init__()

		self.num_filters_total = num_filters * len(filter_sizes)
		self.W = nn.Parameter(torch.empty(vocab_size, embedding_size).uniform_(-1, 1))
		self.Weight = nn.Parameter(torch.empty(self.num_filters_total, num_classes).uniform_(-1, 1))
		self.Bias = nn.Parameter(0.1 * torch.ones([num_classes]))

	def forward(self, x):
		embedded_chars = self.W[x] # bs,len,emb_size

		embedded_chars = embedded_chars.unsqueeze(1) # bs, 1, len, emb_size

		pooled_outputs = []

		for filter_size in filter_sizes:
			conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size), bias=True)(embedded_chars)


			h = F.relu(conv)

			mp = nn.MaxPool2d((sequence_length - filter_size + 1, 1))

			pooled = mp(h)
			pooled = pooled.permute(0, 3, 2, 1)

			pooled_outputs.append(pooled)

		h_pool = torch.cat(pooled_outputs, dim=len(filter_sizes))

		h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])

		out = torch.mm(h_pool_flat, self.Weight) + self.Bias

		return out

if __name__ == '__main__':
	model = TextCNN()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	for e in range(1000):
		output = model(input_batch)
		loss = criterion(output, target_batch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if e % 100 == 0:
			print(f'epoch {e}: train loss {loss.item()}')

















