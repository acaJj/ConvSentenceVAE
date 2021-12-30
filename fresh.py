import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):
        super(VAE, self).__init__()

        self.input_size = 32
        self.latent_size = latent_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.encodinglayer1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_size), nn.ReLU()
        )
        self.encodinglayer2_mean = nn.Sequential(nn.Linear(hidden_size, latent_size))
        self.encodinglayer2_logvar = nn.Sequential(nn.Linear(hidden_size, latent_size))
        self.decodinglayer = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.input_size),
            nn.Sigmoid(),
        )

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, length):
        batch_size = x.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        x = x[sorted_idx]

        x = x.view(-1, batch_size)
        x = self.encodinglayer1(x.float())
        log_var = self.encodinglayer2_logvar(x)
        mean = self.encodinglayer2_mean(x)       
        std = torch.exp(0.5 * log_var)

        z = self.sample(log_var, mean)
        x = self.decodinglayer(z)
        #padded_outputs = rnn_utils.pad_packed_sequence(x, batch_first=True)[0]
        #decoder_output = x.view()
        #padded_outputs = padded_outputs.contiguous()
        #_,reversed_idx = torch.sort(sorted_idx)
        #padded_outputs = padded_outputs[reversed_idx]
        b,s = x.size()
        # project outputs to vocab
        logp = nn.functional.log_softmax(x.view(-1, x.size(1)), dim=0)
        logp = x.view(b, s, -1)
        print(logp.shape)

        print(self.pad_idx)
        print(self.sos_idx)
        print(self.eos_idx)
        print(self.unk_idx)
        #z = to_var(torch.randn([batch_size, self.latent_size]))
        #z = z * std + mean

        #print(x)
        #print(mean)
        #print(log_var)

        return logp, mean, log_var, z