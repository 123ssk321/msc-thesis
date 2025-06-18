import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng
from tsl.nn import get_functional_activation
from tsl.nn.blocks import MLPDecoder, RNN
from tsl.nn.layers import GraphConv, TemporalConv


class TimeSpaceModel(nn.Module):
    def __init__(self, time_module: str, space_module: str, input_size: int, output_size: int,
                 hidden_size: int, time_layers: int, space_layers: int, window: int, horizon: int, num_nodes: int,
                 batch_size: int):
        super(TimeSpaceModel, self).__init__()
        self.window = window
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.batch_size = batch_size

        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), )
        self.time_nn = self._time_module(time_module, hidden_size, time_layers)

        self.activation = get_functional_activation('relu')
        # self.dropout = nn.Dropout(0.2)
        graph_convs = []
        if space_module is not None:
            space_module = self._space_module(space_module, hidden_size, space_layers)
            for i in range(space_layers):
                graph_convs.append(space_module)
        self.space_convs = nn.ModuleList(graph_convs)
        self.decoder = MLPDecoder(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  output_size=output_size,
                                  activation='relu',
                                  receptive_field=window,
                                  horizon=horizon)

    def cforward(self, x, edge_index, edge_weight, window_size=None):
        # x: [batch*window*nodes, features]

        # print(f'x: {x.shape}')
        # print(f'window_size: {window_size}')
        x_b = x.view(self.batch_size, int(x.shape[0] / self.batch_size), -1)  # [batch, window*nodes, features]
        # print(f'x_b: {x_b.shape}')
        x_enc = self.input_encoder(x_b)  # linear encoder: x_enc = xΘ + b [batch, window*nodes, hidden_size]
        # print(f'x_enc: {x_enc.shape}')
        current_size = window_size  # x_enc.shape[1]  # 25632
        target_size = self.window  # 8 * 801  # 51264
        padding_size = target_size - current_size  # 25632

        x_enc_b = x_enc.view(self.batch_size, self.window, -1,
                             x_enc.shape[2]) if self.window == window_size else torch.cat(
            [x_enc.view(self.batch_size, window_size, -1, x_enc.shape[2]),
             torch.zeros(self.batch_size, padding_size, 801, x_enc.shape[2]).to(x_enc.device)],
            dim=1)  # .to(x_enc.device)
        # print(f'x_enc_b: {x_enc_b.shape}')  # [batch, window, nodes, hidden_size]
        h = self.time_nn(
            x_enc_b) if self.time_nn is not None else x_enc  # temporal processing: x=[b t n f] -> h=[b t n h]
        # print(f'h: {h.shape}')  # [batch, window, nodes, hidden_size]
        h = h.reshape(self.batch_size * self.window * h.shape[2], -1) if self.window == window_size else h.reshape(
            self.batch_size * self.window * h.shape[2], -1)[:x.shape[0], :]  # [batch*window*nodes, hidden_size]
        # print(f'h_reshaped: {h.shape}')
        z = torch.zeros_like(h).to(h.device)  # [batch*window*nodes, hidden_size]
        # for b in range(h.size(0)):
        #     for t in range(h.size(1)):
        #         if len(self.space_convs) > 0:
        #             for space_conv in self.space_convs :
        #                 # z[b][t] = self.dropout(self.activation(space_conv(h[b][t], edge_index[t], edge_weight[t])))  # spatial processing
        #                 z[b][t] = self.activation(space_conv(h[b][t], edge_index[t], edge_weight[t]))  # spatial processing
        #                 # z[b][t] = self.activation(space_conv(h[b][t], edge_index[t]))  # spatial processing
        for space_conv in self.space_convs:
            z = self.activation(space_conv(h, edge_index, edge_weight))

        # print(f'z: {z.shape}')
        # print(f'padding_size: {padding_size}')
        z_b = z.view(self.batch_size, self.window, -1, z.shape[1]) if self.window == window_size else torch.cat(
            [z.view(self.batch_size, window_size, -1, z.shape[1]),
             torch.zeros(self.batch_size, padding_size, 801, z.shape[1]).to(z.device)], dim=1)  # .to(z.device)
        # print(f'z_b: {z_b.shape}')  # [batch, window, nodes, hidden_size]
        o = self.decoder(z_b)  # output prediction
        # print(f'o: {o.shape}')
        o_b = o.reshape(self.batch_size * self.window * z_b.shape[2], -1) if self.window == window_size else o.reshape(
            self.batch_size * self.window * z_b.shape[2], -1)[:x.shape[0], :]
        # print(f'o_reshaped: {o_b.shape}')  # [batch*window*nodes, output_size]
        return o_b

    def forward(self, input_tensor, edge_index, edge_weight, num_windows_in_batch):
        # input_tensor: [batch_size * time_steps * num_nodes, features]
        # print(f'input_tensor: {input_tensor.shape}')

        reshaped_input = input_tensor.view(self.batch_size, int(input_tensor.shape[0] / self.batch_size),
                                           -1)  # [batch_size, time_steps * num_nodes, features]
        # print(f'reshaped_input: {reshaped_input.shape}')

        encoded_input = self.input_encoder(
            reshaped_input)  # Linear encoder: encoded_input = input_tensor * weights + bias [batch_size, time_steps * num_nodes, hidden_size]
        # print(f'encoded_input: {encoded_input.shape}')

        current_sequence_length = num_windows_in_batch
        target_sequence_length = self.window
        padding_length = target_sequence_length - current_sequence_length
        # print(f'current_sequence_length: {current_sequence_length}')
        # print(f'target_sequence_length: {target_sequence_length}')
        # print(f'padding_length: {padding_length}')

        encoded_input_reshaped = encoded_input.view(self.batch_size, -1, self.num_nodes, encoded_input.shape[
            2])  # [batch_size, time_steps, num_nodes, hidden_size]
        # print(f'num_nodes: {self.num_nodes}')
        if self.batch_size != num_windows_in_batch:
            padding_tensor = torch.zeros(self.batch_size, padding_length, self.num_nodes, encoded_input.shape[2]).to(
                encoded_input.device)
            encoded_input_reshaped = torch.cat([encoded_input_reshaped, padding_tensor], dim=1)

        # print(f'encoded_input_reshaped: {encoded_input_reshaped.shape}')  # [batch_size, time_steps, num_nodes, hidden_size]

        temporal_output = self.time_nn(
            encoded_input_reshaped) if self.time_nn is not None else encoded_input  # Temporal processing
        # print(f'temporal_output: {temporal_output.shape}')  # [batch_size, time_steps, num_nodes, hidden_size]

        reshaped_temporal_output_size = self.batch_size * self.window * self.num_nodes  # because tensor was padded with zeros it is necessary to use self.window instead of time_steps
        temporal_output_reshaped = temporal_output.reshape(reshaped_temporal_output_size, -1)
        if self.batch_size != num_windows_in_batch:
            temporal_output_reshaped = temporal_output_reshaped[:input_tensor.shape[0], :]
        # print(f'temporal_output_reshaped: {temporal_output_reshaped.shape}')

        # TODO: Check if this is correct. For multiple space conv layers, we need to pass the output of previous layer to next layer
        spatial_output = torch.zeros_like(temporal_output_reshaped).to(
            temporal_output_reshaped.device)  # [batch_size * time_steps * num_nodes, hidden_size]
        if len(self.space_convs) > 0:
            for spatial_layer in self.space_convs:
                spatial_output = self.activation(spatial_layer(temporal_output_reshaped, edge_index, edge_weight))
        else:
            spatial_output = temporal_output_reshaped

        # TODO: This is correct solution to above problem
        # spatial_output = temporal_output_reshaped  # [batch_size * time_steps * num_nodes, hidden_size]
        # if len(self.space_convs) > 0:
        #     for spatial_layer in self.space_convs:
        #         spatial_output = self.activation(spatial_layer(temporal_output_reshaped, edge_index, edge_weight))

        # print(f'spatial_output: {spatial_output.shape}')

        spatial_output_reshaped = spatial_output.view(self.batch_size, -1, self.num_nodes, spatial_output.shape[1])
        if self.batch_size != num_windows_in_batch:
            padding_tensor = torch.zeros(self.batch_size, padding_length, self.num_nodes, spatial_output.shape[1]).to(
                spatial_output.device)
            spatial_output_reshaped = torch.cat([spatial_output_reshaped, padding_tensor], dim=1)
        # print(f'spatial_output_reshaped: {spatial_output_reshaped.shape}')  # [batch_size, time_steps, num_nodes, hidden_size]

        decoded_output = self.decoder(spatial_output_reshaped)  # Output prediction
        # print(f'decoded_output: {decoded_output.shape}')  # [batch_size, horizon, num_nodes, output_size]

        reshaped_decoded_output_size = self.batch_size * decoded_output.shape[
            1] * self.num_nodes  # because tensor was padded with zeros it is necessary to use self.window instead of time_steps
        final_output = decoded_output.reshape(reshaped_decoded_output_size, -1)
        if self.batch_size != num_windows_in_batch:
            final_output = final_output[:input_tensor.shape[0], :]
        # print(f'final_output: {final_output.shape}')  # [batch_size * time_steps * num_nodes, output_size]

        return final_output

    def _time_module(self, time_module: str, hidden_size: int, num_layers: int = 1):
        if time_module == 'gru':
            return RNN(input_size=hidden_size,
                       hidden_size=hidden_size,
                       n_layers=num_layers,
                       cell='gru',
                       return_only_last_state=False)
        elif time_module == 'lstm':
            return RNN(input_size=hidden_size,
                       hidden_size=hidden_size,
                       n_layers=num_layers,
                       cell='lstm',
                       return_only_last_state=False)
        elif time_module == 'multigru':
            pass
        elif time_module == 'multilstm':
            pass

        elif time_module == 'graphconvgru':
            pass
        elif time_module == 'graphconvlstm':
            pass
        elif time_module == 'dcrnn':
            pass
        elif time_module == 'evolvegcn':
            pass
        elif time_module == 'tranformer':
            pass
        elif time_module == 'cnn':
            return TemporalConv(input_channels=hidden_size,
                                output_channels=hidden_size,
                                kernel_size=3,
                                stride=1,
                                dilation=1,
                                channel_last=True)
        elif time_module == 'gcnn':
            pass
        elif time_module == 'att':
            pass
        elif time_module == 'tcn':
            pass
        elif time_module == 'condtcn':
            pass
        elif time_module == 'stcn':
            pass
        elif time_module is None:
            return None
        else:
            raise ValueError(f'Unknown time module: {time_module}')

    def _space_module(self, space_module: str, hidden_size: int, num_layers: int = 1):
        if space_module == 'gcn':
            return GraphConv(input_size=hidden_size, output_size=hidden_size, activation='relu')
        elif space_module == 'gat':
            pass
        elif space_module == 'sage':
            return nng.SAGEConv(hidden_size, hidden_size)
        elif space_module == 'diffconv':
            pass
        elif space_module == 'gatconv':
            pass
        elif space_module is None:
            return None
        else:
            raise ValueError(f'Unknown space module: {space_module}')
