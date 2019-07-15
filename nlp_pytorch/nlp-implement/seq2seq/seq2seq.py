import numpy as np
import torch
from torch import nn
from torch.optim import Adam

# start end blank
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
num_dic = {n: i for i, n in enumerate(char_arr)}

seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

n_step = 5
n_hidden = 128
n_class = len(num_dic)
batch_size = len(seq_data)


def make_batch(seq_data):
	input_batch, output_batch, target_batch = [], [], []
	for seq in seq_data:
		for i in range(2):
			seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

		input = [num_dic[n] for n in seq[0]] #seq1
		output = [num_dic[n] for n in ('S' + seq[1])] # seq2
		target = [num_dic[n] for n in (seq[1] + 'E')] # seq3

		input_batch.append(np.eye(n_class)[input])
		output_batch.append(np.eye(n_class)[output])
		target_batch.append(target)

	return torch.Tensor(input_batch), torch.Tensor(output_batch), torch.LongTensor(target_batch)

class Seq2Seq(nn.Module):
	def __init__(self):
		super(Seq2Seq, self).__init__()
		
		self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
		self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
		self.fc = nn.Linear(n_hidden, n_class)


	def forward(self, enc_input, enc_hidden, dec_input):
		enc_input = enc_input.transpose(0, 1)
		dec_input = dec_input.transpose(0, 1)

		_, enc_states = self.enc_cell(enc_input, enc_hidden)

		outputs, _ = self.dec_cell(dec_input, enc_states)

		out = self.fc(outputs)

		return out

def translate(word):
	input_batch, output_batch, _ = make_batch([[word, 'P' * len(word)]])

	hidden = torch.zeros(1, 1, n_hidden)
	output = model(input_batch, hidden, output_batch)

	predict = output.data.max(2, keepdim=True)[1]

	decoded = [char_arr[i] for i in predict]

	end = decoded.index('E')

	translated = ''.join(decoded[:end])

	return translated.replace('P', '')


if __name__ == '__main__':
	model = Seq2Seq()
	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=0.001)

	input_batch,output_batch,target_batch = make_batch(seq_data)

	for e in range(1000):
		hidden = torch.zeros(1, batch_size, n_hidden)
		output = model(input_batch, hidden, output_batch)
		output = output.transpose(0, 1)

		loss = 0
		for i in range(0, len(target_batch)):
			loss += criterion(output[i], target_batch[i])

		if e % 100 == 0:
			print(f'epoch:{e} -- loss:{loss}')

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()



	print('man ->', translate('man'))
	print('mans ->', translate('mans'))
	print('king ->', translate('king'))
	print('black ->', translate('black'))
	print('upp ->', translate('upp'))













