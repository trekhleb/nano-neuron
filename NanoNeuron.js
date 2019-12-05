// NanoNeuron model.
// It implements basic linear dependency between 'x' and 'y': y = w * x + b.
// Simply saying our NanoNeuron is a kid that can draw the straight line in XY coordinates.
function NanoNeuron(w, b) {
  // NanoNeuron knows only about these two parameters of linear function.
  // These parameters are something that NanoNeuron is going to "learn" during the training process.
  this.w = w;
  this.b = b;
  // This is the only thing that NanoNeuron can do - imitate linear dependency.
  // It accepts some input 'x' and predict the output 'y'. No magic here.
  this.predict = (x) => {
    return x * this.w + this.b;
  }
}

// Convert Celsius values to Fahrenheit using formula: f = 1.8 * c + 32.
// Ultimately we want to teach our NanoNeuron to imitate this function (to learn
// that w = 1.8 and b = 32) without knowing these parameters in advance.
function celsiusToFahrenheit(c) {
  const w = 1.8;
  const b = 32;
  const f = c * w + b;
  return f;
};

// Generate training examples.
// We will use this data to train our NanoNeuron.
// Before our NanoNeuron will grow and will be able to make decisions by its own
// we need to teach it what is right and what is wrong using training examples. 
function generateTrainingSet() {
  const xTrain = [];
  const yTrain = [];
  for (let x = 0; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTrain.push(x);
    yTrain.push(y);
  }
  return [xTrain, yTrain];
}

// Generate test examples.
// This data will be used to evaluate how well our NanoNeuron performs on the data
// that it didn't see during the training. This is the point where we could
// see that our kid has grown and can make decisions on its own.
function generateTestSet() {
  const xTest = [];
  const yTest = [];
  // By starting from 0.5 and using the same step 1 as we have used for training set
  // we make sure that test set has different data comparing to training set.
  for (let x = 0.5; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTest.push(x);
    yTest.push(y);
  }
  return [xTest, yTest];
}

// Calculate the cost (the mistake) between the correct value of 'y' and 'prediction' that NanoNeuron made.
function predictionCost(y, prediction) {
  // This is a simple difference between two values.
  // The closer the values to each other - the smaller the difference.
  // We're using power of 2 here just to get rid of negative numbers
  // so that (1 - 2) ^ 2 would be the same as (2 - 1) ^ 2.
  // Division by 2 is happening just to simplify further backward propagation formula (see below).
  return (y - prediction) ** 2 / 2;
}

// Forward propagation.
// This function takes all examples from training sets xTrain and yTrain and calculates
// model predicts for each example from xTrain.
// Along the way it also calculates the prediction cost.
function forwardPropagation(model, xTrain, yTrain) {
  const m = xTrain.length;
  const predictions = [];
  let cost = 0;
  for (let i = 0; i < m; i += 1) {
    const prediction = nanoNeuron.predict(xTrain[i]);
    cost += predictionCost(yTrain[i], prediction);
    predictions.push(prediction);
  }
  cost /= m;
  return [cost, predictions];
}

// Backward propagation.
// This is the place where machine learning mistakenly looks like a magic.
// The key concept here is derivative which shows what step to take to get closer
// to function minimum. Remember, finding the minimum of a cost function is the
// ultimate goal of training process. The cost function looks like this:
// (y - prediction) ^ 2 * 1/2, where prediction = x * w + b.
function backwardPropagation(predictions, xTrain, yTrain) {
  const m = xTrain.length;
  let dW = 0;
  let dB = 0;
  for (let i = 0; i < m; i += 1) {
    // This is derivative of the cost function by `w` param.
    dW += (yTrain[i] - predictions[i]) * xTrain[i];
    // This is derivative of the cost function by `b` param.
    dB += yTrain[i] - predictions[i];
  }
  dW /= m;
  dB /= m;
  return [dW, dB];
}

// Train the model.
// This is a teacher for our NanoNeuron model:
// - it will spend some time (epochs) with our yet stupid NanoNeuron model and try to train/teach 
// - it will use specific books (xTrain and yTrain dataset) for training
// - it will push our kid to learn harder or softer based on a learning rate 'alpha'
// (the harder the push the faster our nano-kid will learn but if teacher will push too hard the kid will have a nervous breakdown)
function trainModel({model, epochs, alpha, xTrain, yTrain}) {
  // The is the history of how NanoNeuron has learnt. It might have a good or bad marks during the learning 
  const costHistory = [];

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    // Forward propagation for all training examples.
    // Let's save the cost for current iteration.
    // This will help us to analyse how our model learns.
    const [cost, predictions] = forwardPropagation(model, xTrain, yTrain);
    costHistory.push(cost);
  
    // Backward propagation. Let's learn some lessons from the mistakes.
    // This function returns smalls steps we need to take for for params 'w' and 'b'
    // to make predictions more accurate.
    const [dW, dB] = backwardPropagation(predictions, xTrain, yTrain);
  
    // Adjust our NanoNeuron parameters to increase accuracy of our model predictions.
    nanoNeuron.w += alpha * dW;
    nanoNeuron.b += alpha * dB;
  
    // console.log({
    //   cost,
    //   w: nanoNeuron.w,
    //   b: nanoNeuron.b,
    // });
  }

  // Let's return cost history from the function to be able to log or to plot it.
  return costHistory;
}

// Now let's use the functions we have created above.

// Let's create our NanoNeuron instance.
// At this moment NanoNeuron doesn't know what values should be set for parameters 'w' and 'b'.
// So let's set up 'w' and 'b' randomly.
const w = Math.random(); // i.e. -> 0.9492
const b = Math.random(); // i.e. -> 0.4570
const nanoNeuron = new NanoNeuron(w, b);

// Generate training and test data-sets.
const [xTrain, yTrain] = generateTrainingSet(); // xTrain -> [0, 1, 2, ...], yTrain -> [32, 33.8, 35.6, ...]
const [xTest, yTest] = generateTestSet(); // xTest -> [0.5, 1.5, 2.5, ...], yTest -> [32.9, 34.7, 36.5, ...]

console.log(yTest);

// Let's train the model with small 0.0005 steps during the 70000 epochs.
// You can play with these parameters, they are being defined empirically.
const costHistory = trainModel({
  model: nanoNeuron,
  epochs: 70000,
  alpha: 0.0005,
  xTrain,
  yTrain,
});

// Evaluate our model accuracy for test dataset to see how well our NanoNeuron 
// deals with new unknown data predictions.
