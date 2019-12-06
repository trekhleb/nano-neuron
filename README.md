# NanoNeuron

> 7 simple JavaScript functions that will give you a feeling of how machines can actually "learn".

## TL;DR

[NanoNeuron](https://github.com/trekhleb/nano-neuron) is _over-simplified_ version of a Neuron concept from the Neural Networks. NanoNeuron is trained to convert a temperature values from Celsius to Fahrenheit.

[NanoNeuron.js](https://github.com/trekhleb/nano-neuron/blob/master/NanoNeuron.js) code example contains 7 simple JavaScript functions (model prediction, cost calculation, forward and backwards propagation, training) that will give you a feeling of how machines can actually "learn". No 3rd-party libraries, no external data-sets and dependencies, only pure and simple JavaScript functions.

☝🏻These functions by any means are **NOT** a complete guide to machine learning. A lot of machine learning concepts are skipped and over-simplified there! This simplification is done in purpose to give the reader a really **basic** understanding and feeling of how machines can learn and ultimately to make it possible for the reader to call it not a "machine learning MAGIC" but rather "machine learning MATH" 🤓.

## What NanoNeuron will learn

You've probably heard about Neurons in the context of [Neural Networks](https://en.wikipedia.org/wiki/Neural_network). NanoNeuron that we're going to implement below is kind of it but simpler. For simplicity reasons we're not even going to build a network on NanoNeurons. We will have it all by itself, alone, doing some magic predictions for us. Namely we will teach this one simple NanoNeuron to convert (predict) the temperature from Celsius to Fahrenheit.

By the way the formula for converting Celsius to Fahrenheit is this:

![Celsius to Fahrenheit](https://github.com/trekhleb/nano-neuron/blob/master/assets/01_celsius_to_fahrenheit.png?raw=true)

### NanoNeuron model

Let's implement our NanoNeuron model function. It implements basic linear dependency between `x` and `y`: `y = w * x + b`. Simply saying our NanoNeuron is a "kid" that can draw the straight line in `XY` coordinates.

`w`, `b` - parameters of the model.

```javascript
function NanoNeuron(w, b) {
  // NanoNeuron knows only about these two parameters of linear function.
  // These parameters are something that NanoNeuron is going to "learn" during the training process.
  this.w = w;
  this.b = b;
  // This is the only thing that NanoNeuron can do - imitate linear dependency.
  // It accepts some input 'x' and predicts the output 'y'. No magic here.
  this.predict = (x) => {
    return x * this.w + this.b;
  }
}
```

## Skipped machine learning concepts

- Training set split 70/30.
- Input normalization.
- Vectorized implementation instead of 'for' loop.
- Activation function.
- No local optimum. Use logarithm.
