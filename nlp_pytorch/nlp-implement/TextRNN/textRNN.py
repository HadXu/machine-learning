import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(sorted(word_list)))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)


batch_size = len(sentences)
n_step = 2
n_hidden = 5

def make_batch(sentences):
	input_batch = []
	target_batch = []

	for sen in sentences:
		word = sen.split()
		input = [word_dict[n] for n in word[:-1]]
		target = word_dict[word[-1]]

		input_batch.append(np.eye(n_class)[input])
		target_batch.append(target)

	return torch.Tensor(input_batch), torch.LongTensor(target_batch)



class TextRNN(nn.Module):
	def __init__(self):
		super(TextRNN, self).__init__()
		
		self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
		self.W = nn.Parameter(torch.randn((n_hidden, n_class)))
		self.b = nn.Parameter(torch.randn(n_class))

	def forward(self, hidden, x):
		# bs, len, v
		X = x.transpose(0, 1)
		outputs, hidden = self.rnn(X, hidden)
		outputs = outputs[-1]
		model = torch.mm(outputs, self.W) + self.b
		return model

if __name__ == '__main__':
	model = TextRNN()

	input_batch, target_batch = make_batch(sentences)


	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	for e in range(10000):
		h = torch.zeros(1, batch_size, n_hidden)
		output = model(h, input_batch)

		loss = criterion(output, target_batch)

		if e % 1000 == 0:
			print(f'epoch:{e} -- loss:{loss}')

		loss.backward()
		optimizer.step()














