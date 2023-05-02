
////////////IMPORT TRANING DATA
import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js';
//Input feature pairs (House size, number of bedrooms)
const INPUTS = TRAINING_DATA.inputs;
//Current listed house prices in dollars given their features above
const OUTPUTS = TRAINING_DATA.outputs;;
//Shuffle the two arrays in the same way so inputs still match outputs indexes
tf.util.shuffleCombo(INPUTS, OUTPUTS)

////////CREATE TENSORS
const INPUTS_TENSOR = tf.tensor2d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

//Normalise Tensors
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log('Normalized Values:')
FEATURE_RESULTS.NORMALIZED_VALUES.print();
console.log('Min Values:')
FEATURE_RESULTS.MIN_VALUES.print();
console.log('Max values:')
FEATURE_RESULTS.MAX_VALUES.print();

//Dispose of unneeded Tensors.
INPUTS_TENSOR.dispose();

//Normalize inputs to 0..1
function normalize(tensor, min, max) {
	const result = tf.tidy(function() {
		//Find min value
		const MIN_VALUES = min || tf.min(tensor, 0);
		//Find max value
		const MAX_VALUES = max || tf.max(tensor, 0);

		//Subtract min value from every value
		const TENSOR_SUBTRACT_MIN_VALUES = tf.sub(tensor, MIN_VALUES);

		//Range size of values
		const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

		//Calculate adjusted valyes by dividing each value by range size
		const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUES, RANGE_SIZE);

		return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
	});
	return result;
}

///////CREATE MODEL ARCHITECTURE
const model = tf.sequential();

//one dense layer, 1 neuron and input of 2 features
model.add(tf.layers.dense({
	inputShape: [2],
	units: 1,
}));

model.summary();

train();

async function train() {
	const LEARNING_RATE = 0.01;

	//Compile model with the learning rate and specify a loss function to use
	model.compile({
		optimizer: tf.train.sgd(LEARNING_RATE),
		loss: 'meanSquaredError'
	});

	//Do the training
	let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
		validationSplit: 0.15, //Use 15% of data for validation
		shuffle: true, //Ensure data is shuffled
		batchSize: 32, //Use batch sizes of 64
		epochs: 20 //Go over the data 10 times
	});

	OUTPUTS_TENSOR.dispose();
	FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

	console.log('Average error loss: ' + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
	console.log('Average validation error loss: ' + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));

	evaluate();
}

function evaluate() {
	//predict answer for a single piece of data
	tf.tidy(function() {
		let newInput = normalize(tf.tensor2d([[750, 1]]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES
		);
		let output = model.predict(newInput.NORMALIZED_VALUES);
		output.print();
	});

	FEATURE_RESULTS.MIN_VALUES.dispose();
	FEATURE_RESULTS.MAX_VALUES.dispose();
	model.dispose();

	console.log(tf.memory().numTensors);
}