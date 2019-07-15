import torch
from torch import nn
import numpy as np
from torch import optim
import matplotlib.pyplot as plt

sentences = ["i like dog", "i like cat", "i like animal",
			"dog cat animal", "apple cat dog like", "dog fish milk like",
			"dog cat eyes like", "i like apple", "apple i hate",
			"apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}

print(word_dict)
print(word_sequence)

bs = 20
embedding_size = 2
voc_size = len(word_list)

def random_batch(data, size):
	random_inputs = []
	random_labels = []
	random_index = np.random.choice(range(len(data)), size, replace=False)

	for i in random_index:
		random_inputs.append(np.eye(voc_size)[data[i][0]])
		random_labels.append(data[i][1])
	return torch.Tensor(random_inputs), torch.LongTensor(random_labels)


skip_grams = []
for i in range(1, len(word_sequence) - 1):
	target = word_dict[word_sequence[i]]
	context = [word_dict[word_sequence[i-1]], word_dict[word_sequence[i + 1]]]

	for w in context:
		skip_grams.append([target, w])




class Word2Vec(nn.Module):
	def __init__(self):
		super(Word2Vec, self).__init__()
		self.W = nn.Parameter(-2*torch.rand(voc_size, embedding_size) + 1)
		self.WT = nn.Parameter(-2*torch.rand(embedding_size, voc_size) + 1)

	def forward(self, X):
		h = torch.matmul(X, self.W)
		o = torch.matmul(h, self.WT)
		return o

if __name__ == '__main__':
	model = Word2Vec()
	cirterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	for e in range(10000):
		input_batch, target_batch = random_batch(skip_grams, bs)

		optimizer.zero_grad()
		output = model(input_batch)

		loss = cirterion(output, target_batch)

		if e % 500 == 0:
			print(f'epoch:{e} -- loss:{loss.item()}')

		loss.backward()
		optimizer.step()

	for i, label in enumerate(word_dict):
		W, WT = model.parameters()
		x,y = float(W[i][0]), float(W[i][1])

		plt.scatter(x,y)

		plt.annotate(label, xy=(x,y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

	plt.show()






































