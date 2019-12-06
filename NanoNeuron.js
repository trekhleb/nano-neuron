// NanoNeuron model.
// It implements basic linear dependency between 'x' and 'y': y = w * x + b.
// Simply saying our NanoNeuron is a "kid" that can draw the straight line in XY coordinates.
// w, b - parameters of the model.
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

// Convert Celsius values to Fahrenheit using formula: f = 1.8 * c + 32.
// Ultimately we want to teach our NanoNeuron to imitate this function (to learn
// that w = 1.8 and b = 32) without knowing these parameters in advance.
// c - temperature in Celsius
// f - calculated temperature in Fahrenheit
function celsiusToFahrenheit(c) {
  const w = 1.8;
  const b = 32;
  const f = c * w + b;
  return f;
};

// Generate training and test data-sets based on celsiusToFahrenheit function.
// Data-sets consist of pairs of input values and correctly labeled output values.
// In real life in most of the cases this data would be rather collected than generated.
// For example we might have a set of images of hand-drawn numbers and corresponding set
// of numbers that explain what number is written on each picture.  
function generateDataSets() {
  // Generate TRAINING examples.
  // We will use this data to train our NanoNeuron.
  // Before our NanoNeuron will grow and will be able to make decisions by its own
  // we need to teach it what is right and what is wrong using training examples.
  // xTrain -> [0, 1, 2, ...],
  // yTrain -> [32, 33.8, 35.6, ...]
  const xTrain = [];
  const yTrain = [];
  for (let x = 0; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTrain.push(x);
    yTrain.push(y);
  }

  // Generate TEST examples.
  // This data will be used to evaluate how well our NanoNeuron performs on the data
  // that it didn't see during the training. This is the point where we could
  // see that our "kid" has grown and can make decisions on its own.
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

// Calculate the cost (the mistake) between the correct output value of 'y' and 'prediction' that NanoNeuron made.
function predictionCost(y, prediction) {
  // This is a simple difference between two values.
  // The closer the values to each other - the smaller the difference.
  // We're using power of 2 here just to get rid of negative numbers
  // so that (1 - 2) ^ 2 would be the same as (2 - 1) ^ 2.
  // Division by 2 is happening just to simplify further backward propagation formula (see below).
  return (y - prediction) ** 2 / 2; // i.e. -> 235.6
}

// Forward propagation.
// This function takes all examples from training sets xTrain and yTrain and calculates
// model predictions for each example from xTrain.
// Along the way it also calculates the prediction cost (average error our NanoNeuron made while predicting).
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

// Backward propagation.
// This is the place where machine learning looks like a magic.
// The key concept here is derivative which shows what step to take to get closer
// to the function minimum. Remember, finding the minimum of a cost function is the
// ultimate goal of training process. The cost function looks like this:
// (y - prediction) ^ 2 * 1/2, where prediction = x * w + b.
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

// Train the model.
// This is like a "teacher" for our NanoNeuron model:
// - it will spend some time (epochs) with our yet stupid NanoNeuron model and try to train/teach it, 
// - it will use specific "books" (xTrain and yTrain data-sets) for training,
// - it will push our kid to learn harder (faster) by using a learning rate parameter 'alpha'
//   (the harder the push the faster our "nano-kid" will learn but if the teacher will push too hard 
//    the "kid" will have a nervous breakdown and won't be able to learn anything).
function trainModel({model, epochs, alpha, xTrain, yTrain}) {
  // The is the history array of how NanoNeuron learns.
  // It might have a good or bad "marks" (costs) during the learning process.
  const costHistory = [];

  // Let's start counting epochs.
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    // Forward propagation for all training examples.
    // Let's save the cost for current iteration.
    // This will help us to analyse how our model learns.
    const [predictions, cost] = forwardPropagation(model, xTrain, yTrain);
    costHistory.push(cost);
  
    // Backward propagation. Let's learn some lessons from the mistakes.
    // This function returns smalls steps we need to take for params 'w' and 'b'
    // to make predictions more accurate.
    const [dW, dB] = backwardPropagation(predictions, xTrain, yTrain);
  
    // Adjust our NanoNeuron parameters to increase accuracy of our model predictions.
    nanoNeuron.w += alpha * dW;
    nanoNeuron.b += alpha * dB;
  }

  // Let's return cost history from the function to be able to log or to plot it after training.
  return costHistory;
}

// ===========================================================================================
// Now let's use the functions we have created above.

// Let's create our NanoNeuron model instance.
// At this moment NanoNeuron doesn't know what values should be set for parameters 'w' and 'b'.
// So let's set up 'w' and 'b' randomly.
const w = Math.random(); // i.e. -> 0.9492
const b = Math.random(); // i.e. -> 0.4570
const nanoNeuron = new NanoNeuron(w, b);

// Generate training and test data-sets.
const [xTrain, yTrain, xTest, yTest] = generateDataSets();

// Let's train the model with small (0.0005) steps during the 70000 epochs.
// You can play with these parameters, they are being defined empirically.
const epochs = 70000;
const alpha = 0.0005;
const trainingCostHistory = trainModel({model: nanoNeuron, epochs, alpha, xTrain, yTrain});

// Let's check how the cost function was changing during the training.
// We're expecting that the cost after the training should be much lower than before.
// This would mean that NanoNeuron got smarter. The opposite is also possible. 
console.log('Cost before the training:', trainingCostHistory[0]); // i.e. -> 4694.3335043
console.log('Cost after the training:', trainingCostHistory[epochs - 1]); // i.e. -> 0.0000024

// Let's take a look at NanoNeuron parameters to see what it has learned.
// We expect that NanoNeuron parameters 'w' and 'b' to be similar to ones we have in
// celsiusToFahrenheit() function (w = 1.8 and b = 32) since our NanoNeuron tried to imitate it.
console.log('NanoNeuron parameters:', {w: nanoNeuron.w, b: nanoNeuron.b}); // i.e. -> {w: 1.8, b: 31.99}

// Evaluate our model accuracy for test data-set to see how well our NanoNeuron deals with new unknown data predictions.
// The cost of predictions on test sets is expected to be be close to the training cost.
// This would mean that NanoNeuron performs well on known and unknown data.
[testPredictions, testCost] = forwardPropagation(nanoNeuron, xTest, yTest);
console.log('Cost on new testing data:', testCost); // i.e. -> 0.0000023

// Now, since we see that our NanoNeuron "kid" has performed well in the "school" during the training
// and that he can convert Celsius to Fahrenheit temperatures correctly even for the data it hasn't seen
// we can call it "smart" and ask him some questions. This was the ultimate goal of whole training process.
const tempInCelsius = 70;
const customPrediction = nanoNeuron.predict(tempInCelsius);
console.log(`NanoNeuron "thinks" that ${tempInCelsius}°C in Fahrenheit is:`, customPrediction); // -> 158.0002
console.log('Correct answer is:', celsiusToFahrenheit(tempInCelsius)); // -> 158

// So close! As all the humans our NanoNeuron is good but not ideal :)
// Happy learning to you!
