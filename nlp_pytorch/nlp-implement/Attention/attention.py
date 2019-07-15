import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt




embedding_dim = 2
n_hidden = 5
num_classes = 2

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

input_batch = torch.LongTensor(inputs)
target_batch = torch.LongTensor(targets)

class BiLSTM_Attention(nn.Module):
	def __init__(self):
		super(BiLSTM_Attention, self).__init__()

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)

		self.out = nn.Linear(n_hidden * 2, 2)

	def attention(self, output, final_state):
		hidden = final_state.view(-1, n_hidden * 2, 1)

		attn_weights = torch.bmm(output, hidden).squeeze(2) # bs, n_step

		soft = F.softmax(attn_weights, 1)

		context = torch.bmm(output.transpose(1,2), soft.unsqueeze(2)).squeeze(2)

		return context, soft.data.numpy()




	def forward(self, x):
		input = self.embedding(x) # bs, 3, 2
		input = input.permute(1,0,2)
		hidden_state = torch.zeros(2, len(x), n_hidden)
		cell_state = torch.zeros(2,len(x), n_hidden)

		out, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))

		out = out.permute(1,0,2) # bs, 3, 2

		atten_output, attention = self.attention(out, final_hidden_state)

		return self.out(atten_output), attention

if __name__ == '__main__':
	model = BiLSTM_Attention()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	for e in range(1000):
		output, attention = model(input_batch)
		loss = criterion(output, target_batch)

		if e % 100 == 0:
			print(f'epoch:{e}--loss:{loss.item()}')

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	fig = plt.figure(figsize=(6, 3))
	ax = fig.add_subplot(1, 1, 1)
	ax.matshow(attention, cmap='viridis')

	ax.set_xticklabels(['']+['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
	ax.set_yticklabels(['']+['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'], fontdict={'fontsize': 14})
	plt.show()



































