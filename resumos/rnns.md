# Recurrent Neural Networks

These are my notes for these very popular Networks that are usually employed when we have *sequential data*.

## Recurrent Neurons

Recurrent neurons differ from normal neurons in fundamental way: the output feature vector of a recurrent neuron are fed into the same neuron in a later time step. This enables and makes the recurrent neurons uniquely apt to deal with sequential data, as the data that is fed to these units are three dimensional, where the dimensions are the number of samples $N$, the number of features $D$ and the number of time steps of each feature $T$. In Tensorflow, the data that is fed into the Recurrent Layers are of shape $N \times T \times D$. In the simpler situation where we want to feed a simple time series into the Recurrent Layers, we still need to provide a three dimensional array, where $D=1$, that is, the tensor has shape $N \times T \times 1$. 

### Advantages

One benefit that RNNs have over a traditional ANNs is that the number of parameters needed to handle sequential data is much lower - *is it clear why?*. Having a reduced number of parameters to has a number of benefits. First and foremost, training these special NNs is much more feasible than training the ANN counterparts. ANNs would need a much larger dataset to learn the exploding number of parameters that would arise, so they are much more prone to overfitting than the RNNs which are much more constrained.

## Simple Recurrent Unit

Here follows the update equation for the hidden state, a feature vector with size $M$, the number of hidden units in the recurrent layer: 
$$h_{t} = \tanh(W_{xh}^{T}x_{t} + W_{hh}^{T}h_{t-1} + b_{h})$$

### Disadvantadges

Simple Recurrent Units are particularly vulnerable to the vanishing gradient problem. In particular, input feature vectors that are in the beginning of a sequence are rapidly forgotten due to how deep they are in the formula for the current hidden state vector - *because of recursion*.

## Modern Recurrent Units

Aside from the Simple Recurrent Units we saw previously, there are also two other popular types of recurrent units: the Gated Recurrent Units (GRUs) and the Long-Short Time Memory Units (LSTMs). Although LSTMs have a longer history and have been around since 1997 - GRUs have only been around since 2014 - we will start with the GRUs which can be considered as simplified versions of the LSTMs.

### Gated Recurrent Unit

Here follows the update equations for the hidden state of the Gated Recurrent Units (GRUs):

$$ z_{t} = \sigma(W_{xz}^{T}x_{t} + W_{hz}^{T}h_{t-1} + b_{z}) \newline  r_{t} = \sigma(W_{xr}^{T}x_{t} + W_{hr}^{T}h_{t-1} + b_{r}) \newline  h_{t} = (1-z_{t})\odot h_{t-1} + z_{t}\odot \tanh(W_{xh}^{T}x_{t} + W_{hh}^{T}(r_{t} \odot h_{t-1}) + b_{h}) $$

where $z_{t}$ is the *update gate vector*, $r_{t}$ is the *reset gate vector* and $h_{t} $ is the GRU *hidden state vector*. Recall that the symbol $\odot$ means element-wise multiplication, and that $\sigma$ is the sigmoid function. As for the shapes, all three vectors have size $M$. A simple interpretation is that $z_{t}$ and $r_{t}$ are both *logistic regressions* that produce weights that tells us how much to bring of the *previous* hidden state vector to produce the *current* hidden state vector. In particular, if $z_{t}$ is an M-dimensional array of zeros, the current hidden state is simply previous hidden state. In the case where both $z_{t}$ and $r_{t}$ are arrays of ones, the formula for the current hidden state reduces of the GRU to the formula of the hidden state of a simple recurrent unit. Finally, the reset gate of the responsibility of weighting how much is forgotten from the previous hidden state in the "simple recurrent unit" term of the GRU formula.

Another point to note is that *very* recent research suggest that GRUs actually perform a little worse than LSTMs - when GRUs first came out, that wasn't the case, as research from that time suggested that they performed similarly, something that favored GRUs as they had less parameters.

Questions: *due to the larger amount of parameters, are GRUs more prone to overfitting than simple RNNs?*

### Long-Short Time Memory

LSTMs, as mentioned previously, is more complex than the GRUs, but nevertheless the ideas behind these more powerful models are very similar, as we will see. 

One thing that differentiate the LSTMs from both the GRUs and the simple RNNs is that fact that there is an additional state called the cell state. Most of the time, this cell state will simply be an intermediate vector that is necessary in order to compute the hidden state vector that the LSTM unit outputs.

Bellow we present the equations for the LSTMs, the same way we have done for the other recurrent units:

$$ f_{t} = \sigma(W_{xf}^{T}x_{t} + W_{hf}^{T}h_{t-1} + b_{f}) \newline  i_{t} = \sigma(W_{xi}^{T}x_{t} + W_{hi}^{T}h_{t-1} + b_{i}) \newline
o_{t} = \sigma(W_{xo}^{T}x_{t} + W_{ho}^{T}h_{t-1} + b_{o}) \newline  c_{t} = f_{t}\odot c_{t-1} + i_{t}\odot f_{c}(W_{xc}^{T}x_{t} + W_{hc}^{T} h_{t-1} + b_{c})  \newline
h_{t} = o_{t} \odot f_{h}(c_{t})$$

Although the number of equations is greater and seem more complex, the ideas are very similar to the GRU unit. The vector $f_{t}$, the *forget gate vector* plays a similar role to $1-z_{t}$ in the GRUs, in other words, they dictate how much we forget from the *previous cell state vector* $c_{t-1}$. The vector $i_{t}$, the *input gate vector* in turn, plays a similar role to $z_{t}$. In the LSTMs, there is an additional gate $o_{t}$, the *output gate vector* that dictate how much of the *cell state vector*, the intermediate vector we mentioned previously, is passed on as the next *hidden state*. Finally, in LSTMs, we have these two other functions, $f_{c}$ and $f_{h}$, which are just the activation functions for the *"simple RNN"* term and the new *cell state vector* that will be used to produced the output respectively. In Tensorflow and Keras, these activation functions cannot be controlled individually.

To conclude, we recall that the main advantadge of both GRUs and LSTMs is that they enable long-term memory as they are not as susceptible to the vanishing gradient problem which affects simple RNNs so much.

##### LSTMs with max pooling

One additional setting one may try in order to improve the long-term memory capabilities of LSTMs is adding a MaxPooling layer right after the LSTM unit. What this will do is to take into account all intermediate *hidden state vectors* instead of just considering the last one.