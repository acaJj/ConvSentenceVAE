import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class RCNNSentenceVAE(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification (2015)
    """
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):
    #def __init__(self, vocab_size, embedding_dim, hidden_size, hidden_size_linear, class_num, dropout):
        super(RCNNSentenceVAE, self).__init__()
        #original parameters
        self.embedding_dim = embedding_size
        self.hidden_size_linear = hidden_size
        self.class_num = latent_size # latent_size = 16
        self.dropout = 0

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        #self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        # ENCODER NETWORK
        self.encode_lstm = nn.LSTM(self.embedding_dim, hidden_size, batch_first=True, bidirectional=True, dropout=word_dropout)
        self.encode_W = nn.Linear(self.embedding_dim + 2*hidden_size, self.hidden_size_linear)
        self.encode_tanh = nn.Tanh()
        self.encode_fc = nn.Linear(self.hidden_size_linear, self.class_num)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.hidden2mean = nn.Linear(self.hidden_size_linear, self.class_num) #encoder
        self.hidden2logv = nn.Linear(self.hidden_size_linear, self.class_num) #encoder

        #DECODER NETWORK
        self.decode_lstm = nn.LSTM(self.embedding_dim, hidden_size, batch_first=True, bidirectional=True, dropout=self.dropout)
        self.decode_W = nn.Linear(self.embedding_dim + 2*hidden_size, self.hidden_size_linear)
        self.decode_tanh = nn.Tanh()
        self.decode_fc = nn.Linear(self.hidden_size_linear, self.class_num)
        self.latent2hidden = nn.Linear(self.class_num, hidden_size * self.hidden_factor) #decoder
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size) # decoder + inference method

    def forward(self, x, length):

        #x = torch.Tensor.long(x)
        batch_size = x.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = x[sorted_idx]
        print(batch_size)
        # ENCODING
        # x = |bs, seq_len|
        x_emb = self.embedding(x)
        print(x_emb.shape)
        #print(x_emb)
        packed_input = rnn_utils.pack_padded_sequence(x_emb, sorted_lengths.data.tolist(), batch_first=True)
        # x_emb = |bs, seq_len, embedding_dim|
        output,hidden_states = self.encode_lstm(packed_input)
        #print(output.dtype)
        print("hidden_states Type: " + str(type(output[0])))
        print("x_embedding Type: " + str(type(x_emb)))
        print("hidden_states shape: " + str(output[0].shape))
        print("x_emb shape: " + str(x_emb.shape))
        
        #print(output)
        # output = |bs, seq_len, 2*hidden_size|
        output = torch.cat([output[0].view(32,60), x_emb], 2)
        print(output.shape)
        # output = |bs, seq_len, embedding_dim + 2*hidden_size|
        output = self.encode_tanh(self.encode_W(output)).transpose(1, 2)
        print(output.shape)
        # output = |bs, seq_len, hidden_size_linear| -> |bs, hidden_size_linear, seq_len|
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        print(output.shape)
        # output = |bs, hidden_size_linear|
        output = self.encode_fc(output)
        print(output.shape)
        # output = |bs, class_num|
        print(output.shape)
        #print(output)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            output = output.view(batch_size, self.hidden_size_linear*self.hidden_factor)
        else:
            output = output.squeeze()

        print(output.shape)
        # REPARAMETERIZATION
        #mean = self.hidden2mean(output)
        mean = nn.functional.embedding_bag(output)
        #logv = self.hidden2logv(output)
        logv = torch.log(output)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.class_num]))
        z = z * std + mean

        # DECODING
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(x.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(x.data - self.sos_idx) * (x.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = x.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            x_emb = self.embedding(decoder_input_sequence)
        x_emb = self.embedding_dropout(x_emb)
        #packed_input = rnn_utils.pack_padded_sequence(x_emb, sorted_lengths.data.tolist(), batch_first=True)

        # Run through decoder
        outputs,_ = self.decode_lstm(x_emb)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(outputs.view(-1, outputs.size(2))), dim=-1)
        print("LogP: ")
        print(logp)
        print("\n")
        print("Mean: ")
        print(mean)
        print("\n")
        print("LogV: ")
        print(logv)
        print("\n")
        print("Z: ")
        print(z)
        print("\n")
        #logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        
        #return output
        return logp, mean, logv, z