import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js';

//Input feature pairs (House size, number of bedrooms)
const INPUTS = TRAINING_DATA.inputs;

//Current listed house prices in dollars given their features above
const OUTPUTS = TRAINING_DATA.outputs;;

//Shuffle the two arrays in the same way so inputs still match outputs indexes
tf.util.shuffleCombo(INPUTS, OUTPUTS)

//Create Tensors
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