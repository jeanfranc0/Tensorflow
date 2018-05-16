import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_images(images, img_shape, cls_true, cls_pred = None):
	assert len(images) == len(cls_true) == 9

	# create figure with 3x3 subplots
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace = 3.0, wspace=3.0)

	for i, ax in enumerate(axes.flat):
		# Plot image
		ax.imshow(images[i].reshape(img_shape), cmap='binary')

		# Show true and predicted classes
		if cls_pred is None:
			xlabel = "True : {0}".format(cls_true[i])
		else:
			xlabel = "True : {0}, Pred {1}".format(cls_true[i], cls_pred[i])

		# show the classes as the label on the x-axis
		ax.set_xlabel(xlabel)

		# remove ticks from the plot 
		ax.set_xticks([])
		ax.set_yticks([])

	# plot
	plt.show()



def plot_2D_images(images, cls_true, cls_pred = None):
	assert len(images) == len(cls_true)

	# create figure with 3x3 subplots
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace = 3.0, wspace=3.0)

	for i, ax in enumerate(axes.flat):
		# Plot image
		ax.imshow(images[i], cmap='binary')

		# Show true and predicted classes
		if cls_pred is None:
			xlabel = "True : {0}".format(cls_true[i])
		else:
			xlabel = "True : {0}, Pred {1}".format(cls_true[i], cls_pred[i])

		# show the classes as the label on the x-axis
		ax.set_xlabel(xlabel)

		# remove ticks from the plot 
		ax.set_xticks([])
		ax.set_yticks([])

	# plot
	plt.show()


# ***************** HELPER FUNCTIONS *************************

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.5, shape=[length]))


def batch_norm(x, n_out, is_training):
	"""
	Batch normalization on convolutional maps.
	Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
	Args:
	    x:           Tensor, 4D BHWD input maps
	    n_out:       integer, depth of input maps
	    is_training: boolean tf.Varialbe, true indicates training phase
	    scope:       string, variable scope
	Return:
	    normed:      batch-normalized maps
	"""
	with tf.variable_scope('bn'):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(is_training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed


'''
def batch_norm(inputs, is_training):
	# http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
	decay = 0.999
	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	if is_training:
		batch_mean, batch_var = tf.nn.moments(inputs,[0])
		train_mean = tf.assign(pop_mean,
			pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var,
			pop_var * decay + batch_var * (1 - decay))
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs,
				batch_mean, batch_var, beta, scale, epsilon)
	else:
		return tf.nn.batch_normalization(inputs,
			pop_mean, pop_var, beta, scale, epsilon)
'''

def new_conv_layer(input,
					num_input_channels,
					filter_size,
					num_filters,
					use_norm = False,
					use_pooling = True
					):
	# shape for the filter weights for convolution (format tensorflow API)
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	
	# create weights
	weights = new_weights(shape = shape)
	
	# create biases
	biases = new_biases(length = num_filters)
	
	# create tensorflow operation for convolution
	# strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the y- and x-axis of the image.

	layer = tf.nn.conv2d(input = input, filter = weights, strides = [1, 1, 1, 1], padding = 'SAME')

    # add biases values to the result of the convolution
	layer += biases
	
	# use batch nomalization
	#if use_norm:
	#	layer = batch_norm(layer, num_filters, is_training = True)

    # use pooling to downsample the images
	if use_pooling:
		# this is 2x2 max pooling, which means that we 
		# consider 2x2 windows and select the largest value
		# in each window. Then, we move 2 pixels to the next window
		layer = tf.nn.max_pool(value = layer,
								ksize = [1, 2, 2, 1],
								strides = [1, 2, 2, 1],
								padding = 'SAME')

    # Rectified Linear Unit (ReLU)
    # it calculates max(x, 0) for each input pixel x
	layer = tf.nn.relu(layer)

	return layer, weights


def flatten_layer(layer):
	# get shape from input layer
	layer_shape = layer.get_shape()

	# The shape of the input layer is assumed to be:
	# layer_shape == [num_images, img_height, img_width, num_channels]

	# the number of features is: img_height * img_width * num_channels
	num_features = layer_shape[1:4].num_elements()

	# reshape the layer to [num_images, num_features]
	# Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
	layer_flat = tf.reshape(layer, [-1, num_features])

	return layer_flat, num_features

 
def new_fc_layer(input, 
				num_inputs,
				num_outputs,
				use_relu = True):
	
	# create new weights and biases
	weights = new_weights(shape = [num_inputs, num_outputs])
	biases = new_biases(length = num_outputs)

	# calculate the layer as the matrix multiplications of
	# the input and weights, and then add the bias

	layer = tf.matmul(input, weights) + biases

	if use_relu:
		layer = tf.nn.relu(layer)

	return layer	


# Split the test-set into smaller batches of this size.
test_batch_size = 16

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)