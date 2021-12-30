import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class ConvVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):
        super(ConvVAE, self).__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.input_size = 32
        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        #embedding_size = 300, hidden_size = 256, latent_size = 16
        self.encodinglayer1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=self.hidden_size, kernel_size=self.num_layers), 
            nn.ReLU(),
        )

        self.padding_layer = nn.ConstantPad1d(2,1.0)

        self.encodinglayer2_mean = nn.Sequential(nn.Linear(self.hidden_size, self.latent_size))
        self.encodinglayer2_logvar = nn.Sequential(nn.Linear(self.hidden_size, self.latent_size))

        self.decoder_rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.decodinglayer = nn.Sequential(
            nn.Conv1d(in_channels = embedding_size, out_channels = self.hidden_size, kernel_size = num_layers),
            #nn.ConvTranspose1d(in_channels = embedding_size, out_channels = self.hidden_size, kernel_size = num_layers),
            nn.ReLU(),
            #nn.Sigmoid(),
        )

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx = pad_idx)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size) #encoder
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size) #encoder
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor) #decoder
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, length):
        batch_size = x.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        #print(x.shape)
        x = x[sorted_idx]
        #print(x.shape)

        #PROCESS ENCODER INPUT
        input_embedding = self.embedding(x)
        #print(input_embedding.shape)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        #input_embedding = input_embedding.view(32,300,60)
        input_data = packed_input.data
        #print(input_data.shape)
        #padded_input = self.padding_layer(input_embedding)
        #print(padded_input.shape)
        input_data = input_data.unsqueeze(1)
        x = self.encodinglayer1(input_data.view(input_data.size(0),input_data.size(2),input_data.size(1)))
        #print(x.shape)
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            x = x.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            x = x.squeeze()
        #x = x.view(-1,self.hidden_size)
        #print("Getting Values:")
        #print(x.shape)

        # Used for Reparameterization
        log_var = self.encodinglayer2_logvar(x)
        mean = self.encodinglayer2_mean(x)       
        std = torch.exp(0.5 * log_var)

        z = to_var(torch.randn([log_var.size(0), self.latent_size]))
        #print(z.size())
        #print(std.size())
        #print(mean.size())
        z = z*std + mean
        
        #z = self.sample(log_var, mean)
        #z = z.view(z.size(0), z.size(1), -1)

        #print("1. " + str(input_embedding.shape))
        
        # PROCESS DECODER INPUT
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # PROCESS DECODER INPUT
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(x.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(x.data - self.sos_idx) * (x.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = x.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence.type(torch.LongTensor))
            #print("1.5. " + str(input_embedding.shape))

        #print("2. " + str(input_embedding.shape))
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        input_data = packed_input.data
        #print("3. " + str(input_embedding.shape))
        #input_embedding = input_embedding.view(-1,self.latent_size,input_embedding.size(2))
        #print("4. " + str(input_embedding.shape))
        #padded_input = self.padding_layer(input_embedding)
        #print("4+. " + str(padded_input.shape))
        input_data = input_data.unsqueeze(1)
        #x = self.decodinglayer(input_data.view(input_data.size(0),input_data.size(2),input_data.size(1)))
        outputs,_ = self.decoder_rnn(packed_input)
        #print("AFTER DECODING: " + str(x.shape))

        #PROCESS DECODER OUTPUT - PROBLEM:sequence length 's' not right
        #padded_output = self.padding_layer(x)
        #packed_input.data = x
        padded_output = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_output = padded_output.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_output = padded_output[reversed_idx]
        b,s,_ = padded_output.size()

        #print(padded_output.shape)
        # project outputs to vocab
        #print("OUTPUTS2VOCAB: "+str((padded_output.view(-1, padded_output.size(1))).size()))
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_output.view(-1, padded_output.size(2))), dim=-1)
        #print(logp.shape)
        logp = logp.view(b, s, self.embedding.num_embeddings) # TRY THIS
        #print("LOGP SHAPE: " + str(logp.shape))

        #print(self.pad_idx)
        #print(self.sos_idx)
        #print(self.eos_idx)
        #print(self.unk_idx)
        #z = to_var(torch.randn([batch_size, self.latent_size]))
        #z = z * std + mean

        #print(x)
        #print(mean)
        #print(log_var)

        return logp, mean, log_var, z


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
