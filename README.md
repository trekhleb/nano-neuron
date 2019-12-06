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

But for now our NanoNeuron doesn't know about it...

### NanoNeuron model

Let's implement our NanoNeuron model function. It implements basic linear dependency between `x` and `y` which looks like `y = w * x + b`. Simply saying our NanoNeuron is a "kid" that can draw the straight line in `XY` coordinates.

Variables `w`, `b` are parameters of the model. NanoNeuron knows only about these two parameters of linear function.
These parameters are something that NanoNeuron is going to "learn" during the training process.

The only thing that NanoNeuron can do is to imitate linear dependency. In its `predict()` method it accepts some input `x` and predicts the output `y`. No magic here.

```javascript
function NanoNeuron(w, b) {
  this.w = w;
  this.b = b;
  this.predict = (x) => {
    return x * this.w + this.b;
  }
}
```

### Celsius to Fahrenheit conversion

The temperature value in Celsius can be converted to Fahrenheit using the following formula: `f = 1.8 * c + 32`, where `c` is a temperature in Celsius and `f` is calculated temperature in Fahrenheit.

```javascript
function celsiusToFahrenheit(c) {
  const w = 1.8;
  const b = 32;
  const f = c * w + b;
  return f;
};
```

Ultimately we want to teach our NanoNeuron to imitate this function (to learn that `w = 1.8` and `b = 32`) without knowing these parameters in advance.

### Generating data-sets

Before the training we need to generate **training** and **test data-sets** based on `celsiusToFahrenheit()` function. Data-sets consist of pairs of input values and correctly labeled output values.

> In real life in most of the cases this data would be rather collected than generated. For example we might have a set of images of hand-drawn numbers and corresponding set of numbers that explain what number is written on each picture.

We will use TRAINING examples data to train our NanoNeuron. Before our NanoNeuron will grow and will be able to make decisions by its own we need to teach it what is right and what is wrong using training examples.

We will use TEST examples to evaluate how well our NanoNeuron performs on the data that it didn't see during the training. This is the point where we could see that our "kid" has grown and can make decisions on its own.

```javascript
function generateDataSets() {
  // xTrain -> [0, 1, 2, ...],
  // yTrain -> [32, 33.8, 35.6, ...]
  const xTrain = [];
  const yTrain = [];
  for (let x = 0; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTrain.push(x);
    yTrain.push(y);
  }

  // xTest -> [0.5, 1.5, 2.5, ...]
  // yTest -> [32.9, 34.7, 36.5, ...]
  const xTest = [];
  const yTest = [];
  // By starting from 0.5 and using the same step of 1 as we have used for training set
  // we make sure that test set has different data comparing to training set.
  for (let x = 0.5; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTest.push(x);
    yTest.push(y);
  }

  return [xTrain, yTrain, xTest, yTest];
}
```

### The cost (the error) of prediction

We need to have some metric that will show how close our model's prediction to correct values. The calculation of the cost (the mistake) between the correct output value of `y` and `prediction` that NanoNeuron made will be made using the following formula:

![Prediction Cost](https://github.com/trekhleb/nano-neuron/blob/master/assets/02_cost_function.png?raw=true)

This is a simple difference between two values. The closer the values to each other the smaller the difference. We're using power of `2` here just to get rid of negative numbers so that `(1 - 2) ^ 2` would be the same as `(2 - 1) ^ 2`. Division by `2` is happening just to simplify further backward propagation formula (see below).

The cost function in this case will be as simple as:

```javascript
function predictionCost(y, prediction) {
  return (y - prediction) ** 2 / 2; // i.e. -> 235.6
}
```

### Forward propagation

To do forward propagation means to do a prediction for all training examples from `xTrain` and `yTrain` data-sets and to calculate the average cost of those prediction along the way.

We just let our NanoNeuron say its opinion at this point, just ask him to guess how to convert the temperature. It might be stupidly wrong here. The average cost will show how wrong our model is right now. This cost value is really valuable since by changing the NanoNeuron parameters `w` and `b` and by doing the forward propagation again we will be able to evaluate if NanoNeuron became smarter or not after parameters changes.

The average cost will be calculated using the following formula:

![Average Cost](https://github.com/trekhleb/nano-neuron/blob/master/assets/03_average_cost_function.png?raw=true)

Here is how we may implement it in code:

```javascript
function forwardPropagation(model, xTrain, yTrain) {
  const m = xTrain.length;
  const predictions = [];
  let cost = 0;
  for (let i = 0; i < m; i += 1) {
    const prediction = nanoNeuron.predict(xTrain[i]);
    cost += predictionCost(yTrain[i], prediction);
    predictions.push(prediction);
  }
  // We are interested in average cost.
  cost /= m;
  return [predictions, cost];
}
```

### Backward propagation

Now when we know how right or wrong our NanoNeuron's predictions are (based on average cost at this point) what should we do to make predictions more precise?

The backward propagation is the answer to this question. Backward propagation is the process of evaluating the cost of prediction and adjusting the NanoNeuron's parameters `w` and `b` so that next predictions would be more precise.

This is the place where machine learning looks like a magic 🧞‍♂️. The key concept here is **derivative** which show what step to take to get closer to the cost function minimum.

Remember, finding the minimum of a cost function is the ultimate goal of training process. If we will find such values of `w` and `b` that our average cost function will be small it would mean that NanoNeuron model does really good and precise predictions.

Derivatives are big separate topic that we will not cover in this article. [MathIsFun](https://www.mathsisfun.com/calculus/derivatives-introduction.html) is a good resource to get a basic understanding of it.

One thing about derivatives that will help you to understand how backward propagation works is that derivative by its meaning is a tangent line to the function curve that points out the direction to the function minimum.

![Derivative slope](https://www.mathsisfun.com/calculus/images/slope-x2-2.svg)

_Image source:_ [_MathIsFun_]([MathIsFun](https://www.mathsisfun.com/calculus/derivatives-introduction.html))

For example on the plot above you see that if we're at the point of `(x=2, y=4)` than the slope tells us to go `left` and `down` to get to function minimum. Also notice that the bigger the slope the faster we should move to the minimum.

```javascript
function backwardPropagation(predictions, xTrain, yTrain) {
  const m = xTrain.length;
  // At the beginning we don't know in which way our parameters 'w' and 'b' need to be changed.
  // Therefore we're setting up the changing steps for each parameters to 0.
  let dW = 0;
  let dB = 0;
  for (let i = 0; i < m; i += 1) {
    // This is derivative of the cost function by 'w' param.
    // It will show in which direction (positive/negative sign of 'dW') and
    // how fast (the absolute value of 'dW') the 'w' param needs to be changed.
    dW += (yTrain[i] - predictions[i]) * xTrain[i];
    // This is derivative of the cost function by 'b' param.
    // It will show in which direction (positive/negative sign of 'dB') and
    // how fast (the absolute value of 'dB') the 'b' param needs to be changed.
    dB += yTrain[i] - predictions[i];
  }
  // We're interested in average deltas for each params.
  dW /= m;
  dB /= m;
  return [dW, dB];
}
```

## Skipped machine learning concepts

- Training set split 70/30.
- Input normalization.
- Vectorized implementation instead of 'for' loop.
- Activation function.
- No local optimum. Use logarithm.
