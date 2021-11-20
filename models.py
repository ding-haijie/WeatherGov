from queue import PriorityQueue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import params


class EncoderGate(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, dropout_p):
        super(EncoderGate, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc_in = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.bi_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.fc_lstm = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.fc_gate = nn.Linear(2 * hidden_dim, 1)

    def _content_selection_gate(self, reply_enc):
        """
        content selection gate (variant of attention mechanism)
        the context of each record can be useful in determining its importance vis-a-vis other records in the table.
        """
        # rep_enc_pre: (batch_size, seq_len, hidden_dim)
        rep_enc_pre = reply_enc.permute(1, 0, 2)
        rep_enc_post = rep_enc_pre.permute(0, 2, 1)
        alpha = F.softmax(torch.matmul(rep_enc_pre, rep_enc_post), dim=2)
        dependencies_enc = torch.matmul(alpha, rep_enc_pre).permute(1, 0, 2)
        content_selection = torch.sigmoid(
            self.fc_gate(torch.cat((reply_enc, dependencies_enc), dim=2)))

        return content_selection

    # noinspection DuplicatedCode
    def forward(self, encoder_input):
        fc_input = self.dropout(self.fc_in(encoder_input))
        encoder_output, (h_n, c_n) = self.bi_lstm(fc_input)
        encoder_output = self.fc_lstm(encoder_output)
        content_selection = self._content_selection_gate(encoder_output)
        encoder_output = torch.relu(
            self.fc_out(torch.cat((encoder_output, fc_input), dim=2)))

        h_n = self.fc_lstm(torch.cat((h_n[-2], h_n[-1]), dim=1))
        h_n = torch.relu(self.fc_out(torch.cat((fc_input[-1], h_n), dim=1).unsqueeze(0)))
        c_n = self.fc_lstm(torch.cat((c_n[-2], c_n[-1]), dim=1))
        c_n = torch.relu(self.fc_out(torch.cat((fc_input[-1], c_n), dim=1).unsqueeze(0)))

        return encoder_output, content_selection, (h_n, c_n)


class DecoderAttn(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, dropout_p):
        super(DecoderAttn, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc_attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + embed_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_output, content_selection):
        """
        # hybrid attention layer #
        each time step of the decoder will determine which parts of the source sequence it will focused on closely.
        based on Bahdanau's paper <Neural Machine Translation by Jointly Learning to Align and Translate>.
        here, the encoder's novel attention will be converged to the decoder's attention to
        get the new attention vector, and this brand-new semantic representation will take both of the attention vector
        into consideration that which record in the source sequence it should focused on.
        """
        repeat_decoder_hidden = decoder_hidden.repeat(
            encoder_output.size(0), 1, 1)
        energy = self.fc_attn(torch.cat((repeat_decoder_hidden, encoder_output), dim=2))
        attn_score = F.softmax(torch.sum(energy, dim=2), dim=0).unsqueeze(2)
        # converge both the encoder's and decoder's attention vector
        attn_with_selector = content_selection * attn_score
        attn_with_selector = attn_with_selector / torch.sum(attn_with_selector, dim=0)
        attn_applied = attn_with_selector * encoder_output
        attn_applied = torch.sum(attn_applied, dim=0).unsqueeze(0)
        return attn_applied

    def forward(self, decoder_input, decoder_hidden, encoder_output, content_selection):
        decoder_input = decoder_input.unsqueeze(0).squeeze(2)
        embed = self.dropout(self.embedding(decoder_input))
        # (hidden_previous, encoder_output, content_selection) needed
        attn_applied = self._weighted_encoder_rep(decoder_hidden[0], encoder_output, content_selection)
        attn_combine = torch.relu(torch.cat((embed, attn_applied), dim=2))
        decoder_output, decoder_hidden = self.lstm(attn_combine, decoder_hidden)
        decoder_output = F.log_softmax(self.fc_out(decoder_output), dim=2)
        return decoder_output, decoder_hidden


class Data2Text(nn.Module):
    def __init__(self, encoder, decoder, beam_width, is_cuda_available):
        super(Data2Text, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beam_width = beam_width
        self.is_cuda_available = is_cuda_available

        self.SOS_TOKEN = 0
        self.EOS_TOKEN = 1
        self.PAD_TOKEN = 2
        self.UNK_TOKEN = 3
        self.max_len = params['max_len']

    def forward(self, seq_input, seq_target, train_mode=True):
        encoder_output, content_selection, encoder_hidden = self.encoder(seq_input)
        decoder_hidden = encoder_hidden
        output_dim = self.decoder.output_dim

        if train_mode:
            batch_size = seq_target.size(1)
            pad_tensor = torch.zeros(output_dim)
            pad_tensor[self.PAD_TOKEN] = 1
            seq_output = pad_tensor.repeat((self.max_len, batch_size, 1)).to(
                'cuda' if self.is_cuda_available else 'cpu')

            for timeStep in range(self.max_len):
                decoder_input = seq_target[timeStep]
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_output, content_selection)
                seq_output[timeStep] = decoder_output.squeeze(0)
        else:
            """
            inference mode - greedy search / beam search 
            beam search with beam_width=1 is equivalent to simple greedy search
            """
            if self.beam_width == 1:  # greedy search
                seq_output = torch.tensor([[self.PAD_TOKEN] for _ in range(self.max_len)]).to(
                    'cuda' if self.is_cuda_available else 'cpu')
                decoder_input = seq_target  # first token is SOS_TOKEN
                for timeStep in range(self.max_len):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_output, content_selection)
                    decoder_input = decoder_output.max(dim=2)[1]
                    seq_output[timeStep] = decoder_input.squeeze(0)
                    # decoder_hidden = detach(decoder_hidden)
                    if decoder_input.item() == self.EOS_TOKEN:
                        break
            else:  # beam search
                seq_output = beam_search(decoder=self.decoder,
                                         decoder_input=seq_target,
                                         decoder_hidden=decoder_hidden,
                                         encoder_output=encoder_output,
                                         content_selection=content_selection)
        return seq_output, content_selection


class BeamSearchNode(object):
    def __init__(self, previous_hidden, previous_node, current_input,
                 inherited_order, prob, length):
        super().__init__()

        self.previous_hidden = previous_hidden
        self.previousNode = previous_node
        self.current_input = current_input
        self.inherited_order = inherited_order
        self.prob = prob
        self.length = length

    def __eq__(self, other):
        return self.inherited_order == other.inherited_order

    def __lt__(self, other):
        return self.inherited_order < other.inherited_order


def beam_search(decoder, decoder_input, decoder_hidden, encoder_output, content_selection):
    max_length = params['max_len']
    beam_width = params['beam_width']
    EOS_TOKEN = 1

    nodes = PriorityQueue()
    next_nodes = PriorityQueue()
    node = BeamSearchNode(previous_hidden=decoder_hidden, previous_node=None,
                          current_input=decoder_input, inherited_order=1,
                          prob=0.0, length=1)
    nodes.put((node.prob, node))

    while True:
        # pop one then put it back or break loop, to check if has arrive EOS_TOKEN or max_length
        eos_prob, eos_node = nodes.get()
        if eos_node.length >= max_length:
            end_node = eos_node
            break
        if eos_node.current_input.item() == EOS_TOKEN and eos_node.previousNode is not None:
            end_node = eos_node
            break
        nodes.put((eos_prob, eos_node))

        qsize = nodes.qsize()
        for idx in range(qsize):
            _, node = nodes.get()
            previous_hidden = node.previous_hidden
            current_input = node.current_input

            decoder_output, decoder_hidden = decoder(
                current_input, previous_hidden, encoder_output, content_selection)
            values, index = decoder_output.data.topk(beam_width)

            for beam in range(beam_width):
                beam_out = index[0, 0, beam].unsqueeze(dim=0).unsqueeze(dim=1)
                beam_prob = -values[0, 0, beam]
                next_node = BeamSearchNode(previous_hidden=decoder_hidden, previous_node=node,
                                           current_input=beam_out,
                                           inherited_order=idx,
                                           prob=node.prob + beam_prob, length=node.length + 1)
                next_nodes.put((next_node.prob, next_node))

        for beam in range(beam_width):
            temp_prob, temp_node = next_nodes.get()
            nodes.put((temp_prob, temp_node))

        next_nodes.queue.clear()  # clear nextNodes queue

    beam_seq2seq_out = end_node.current_input.unsqueeze(dim=0)
    while end_node.previousNode is not None:
        end_node = end_node.previousNode
        beam_seq2seq_out = torch.cat(
            (end_node.current_input.unsqueeze(dim=0), beam_seq2seq_out), dim=0)

    return beam_seq2seq_out.reshape(-1)


def detach(hidden_states):
    """ detach tensor from computational graph """
    if isinstance(hidden_states, tuple):
        return [hidden_state.detach() for hidden_state in hidden_states]
    elif isinstance(hidden_states, Tensor):
        return hidden_states.detach()
    else:
        raise TypeError('wrong type: ', type(hidden_states))
