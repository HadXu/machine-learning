import torch
from torch import nn
import numpy as np
from torch import optim

sentences = ['i like dog', 'i love coffee', 'i hate milk']

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w:i for i, w in enumerate(word_list)}
number_dict = {i:w for i, w in enumerate(word_list)}

n_class = len(word_dict)

def make_batch(sentences):
	input_batch = []
	target_batch = []

	for sen in sentences:
		word = sen.split()
		input = [word_dict[w] for w in word[:-1]]
		target = word_dict[word[-1]]

		input_batch.append(input)
		target_batch.append(target)
	return torch.LongTensor(input_batch), torch.LongTensor(target_batch)

m = 2
n_step = 2
n_hidden = 2

class NNLM(nn.Module):
	def __init__(self):
		super(NNLM, self).__init__()
		self.C = nn.Embedding(n_class, m)
		self.H = nn.Parameter(torch.randn(n_step*m, n_hidden))
		self.W = nn.Parameter(torch.randn(n_step*m, n_class))
		self.d = nn.Parameter(torch.randn(n_hidden))
		self.U = nn.Parameter(torch.randn(n_hidden, n_class))
		self.b = nn.Parameter(torch.randn(n_class))
	def forward(self, x):
		X = self.C(x)
		X = X.view(-1, n_step * m)
		tanh = torch.tanh(self.d + torch.mm(X, self.H))
		out = self.b + torch.mm(X,self.W) + torch.mm(tanh, self.U)
		return out

		


if __name__ == '__main__':
	model = NNLM()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 1e-3)

	input_batch, target_batch = make_batch(sentences)

	print(input_batch.size())
	print(target_batch.size())
	for epoch in range(1000):
		optimizer.zero_grad()
		output = model(input_batch)

		loss = criterion(output, target_batch)

		if epoch % 100 == 0:
			print(f'epoch {epoch}: loss:{loss.item()}')

		loss.backward()
		optimizer.step()




















