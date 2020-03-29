import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from torch.distributions import Categorical, Normal

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


tf = try_import_tf()
torch, nn = try_import_torch()


class TFRNNHybrid(RecurrentTFModelV2):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=128,
                 cell_size=64):
        super(TFRNNHybrid, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.cell_size = cell_size
        # import pdb; pdb.set_trace()
        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense1,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

        # import pdb; pdb.set_trace()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

# PyTorch hybrid RNN models


class RNNHybrid(TorchModelV2, nn.Module):
    """
    RNN model for hybrid actions, i.e., discrete actions with continuous parameters. 

    Example applications: 
        1. Designing multilayer optical structures
        2. ...
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.obs_size = _get_size(obs_space)
        self.hidden_size = model_config["hidden_size"]
        # number of discrete actions
        self.num_category = model_config["num_category"]
        # number of parameters for each action
        self.num_params = model_config['num_params']

        self.rnn = nn.GRUCell(self.obs_size, self.hidden_size)

        # category head (for discrete action)
        self.h2c = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(self.hidden_size, self.num_category))

        # parameter head (for continuous parameters of the action)
        self.h2p = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(self.hidden_size, self.num_params))

        # critic
        self.critic = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden_size, 1))

    @override(TorchModelV2)
    def get_initial_state(self):
        # make hidden states on same device as model
        return [self.critic.weight.new(1, self.hidden_size).zero_().squeeze(0)]

    @override(TorchModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        # x = nn.functional.relu(self.fc1(input_dict["obs_flat"].float()))
        h_in = hidden_state[0].reshape(-1, self.rnn_hidden_dim)
        x = input_dict['obs_flat'].float()

        h = self.rnn(x, h_in)
        c, p = self.h2c(h), self.h2p(h)
        q = torch.cat((c.view(-1, self.num_category),
                       p.view(-1, self.num_params)))
        self._value_out = self.critic(h)
        return q, [h]

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out


class GRU_RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, critic=False):
        super(GRU_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size  # number of materials + 1
        self.input_size = input_size  # number of discrete thicknesses
        self.critic = critic

        self.rnncell = nn.GRUCell(output_size + input_size, hidden_size)

        # category head
        self.o2c = nn.Linear(hidden_size, output_size)

        # thickness head
        self.o2d = nn.Linear(hidden_size, input_size)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input), 1)
        hidden = self.rnncell(input_combined, hidden)

        # material type
        category = self.o2c(hidden)

        # thickness
        d = self.o2d(hidden)

        # we only output probablitiy if the network is used as an actor
        if not self.critic:
            category = self.softmax(category)
            d = self.softmax(d)

        return category, d, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters())
        return weight.new_zeros(batch_size, self.hidden_size)

    def sample(self, max_length, random_choose=False):
        '''
        Samples a sequence from the sequence generative model.
        Input:
            max_length:
            random_choose: if True, then select the actions randomly
        '''

        hidden = self.init_hidden(1)
        category = hidden.new_zeros(self.output_size).view(1, -1)
        # at the start of sequence, we initialize the thickness as 0
        thickness = hidden.new_zeros((1, self.input_size))
        categories = []
        values = []
        log_probs = []

        for i in range(max_length):

            category, thickness, hidden = self.forward(
                category, thickness, hidden)

            # random select
            if random_choose:
                cs = list(range(self.output_size))
                ds = list(range(self.input_size))
                c, d = np.random.choice(cs), np.random.choice(ds)
                log_pc, log_pd = torch.log(
                    category[0, c]), torch.log(thickness[0, d])

            else:
                mc, md = Categorical(category), Categorical(thickness)
                c, d = mc.sample(), md.sample()
                log_pc, log_pd = mc.log_prob(c), md.log_prob(d)
                c, d = c.item(), d.item()
                category, thickness = torch.zeros_like(
                    category), torch.zeros_like(thickness)
                # print(category, thickness)
                category[0, c], thickness[0, d] = 1, 1

            if c == self.output_size - 1:
                # terminate token
                log_probs.append(log_pc)
                # log_probs.append(log_pd)
                break
            else:
                categories.append(c)
                values.append(d)
                log_probs.append(log_pc)
                log_probs.append(log_pd)

        categories, values = np.array(categories), np.array(values)

        return categories, values, log_probs


class RNN(nn.Module):
    '''
    Character-level RNN model adapted from: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
    '''

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size

        self.i2h = nn.Linear(output_size + input_size +
                             hidden_size, hidden_size)
        self.i2o = nn.Linear(output_size + input_size +
                             hidden_size, hidden_size)

        # hidden layer
        self.o2o = nn.Linear(2 * hidden_size, hidden_size)

        # category head
        self.o2c = nn.Linear(hidden_size, output_size)

        # thickness head
        self.o2d = nn.Linear(hidden_size, input_size)

        # self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, category, input, hidden):

        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.relu(self.i2h(input_combined))
        output = self.relu(self.i2o(input_combined))
        output_combined = torch.cat((hidden, output), 1)
        output = self.relu(self.o2o(output_combined))

        category = self.o2c(output)
        category = self.softmax(category)

        d = self.o2d(output)
        d = self.softmax(d)

        return category, d, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters())
        return weight.new_zeros(batch_size, self.hidden_size)

    def sample(self, max_length, random_choose=False):
        '''
        Samples a sequence from the sequence generative model.
        Input:
            max_length:
            random_choose: if True, then select the actions randomly
        '''

        hidden = self.init_hidden(1)
        category = hidden.new_zeros(self.output_size).view(1, -1)
        category[:, 0] = 1
        # at the start of sequence, we initialize the thickness as 0
        thickness = hidden.new_zeros((1, self.input_size))
        categories = []
        values = []
        log_probs = []

        for i in range(max_length):

            category, thickness, hidden = self.forward(
                category, thickness, hidden)

            # random select
            if random_choose:
                cs = list(range(self.output_size))
                ds = list(range(self.input_size))
                c, d = np.random.choice(cs), np.random.choice(ds)
                log_pc, log_pd = torch.log(
                    category[0, c]), torch.log(thickness[0, d])

            else:
                mc, md = Categorical(category), Categorical(thickness)
                c, d = mc.sample(), md.sample()
                log_pc, log_pd = mc.log_prob(c), md.log_prob(d)
                c, d = c.item(), d.item()

            if c == self.output_size - 1:
                break
            else:
                categories.append(c)
                values.append(d)
                log_probs.append(log_pc)
                log_probs.append(log_pd)

        categories, values = np.array(categories), np.array(values)

        return categories, values, log_probs
