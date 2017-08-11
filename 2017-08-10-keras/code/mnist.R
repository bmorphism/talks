library(keras)

# define hyperparameters
batch_size <- 32
num_classes <- 10
epochs <- 25

# input image dimensions
img_rows <- 28
img_cols <- 28

# the data, pre-shuffled and pre-split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape the data for input into the network
# last number is number of color channels in the image array
dim(x_train) <- c(nrow(x_train), img_rows, img_cols, 1)
dim(x_test) <- c(nrow(x_test), img_rows, img_cols, 1)
input_shape <- c(img_rows, img_cols, 1)

# rescale the data so values are between 0 and 1
# (since pixel values are on a 0 to 255 scale)
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# define model
model <- keras_model_sequential()
model %>%
  # convolutional block
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  # pooling for parameter reduction
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # dropout to avoid overfitting
  layer_dropout(rate = 0.25) %>%
  # reduce dimensions for classification
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  # output layer
  layer_dense(units = num_classes, activation = 'softmax')

# we use this so that we don't overtrain,
# which leads to overfitting and wasted compute
stopper <- callback_early_stopping(monitor = "val_loss",
                                   patience = 3,
                                   mode = "min")

# we use this to save the best model from the training run
checker <- callback_model_checkpoint(filepath = "best_model_0.h5",
                                     monitor = "val_loss",
                                     save_best_only = TRUE,
                                     mode = "min")


# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# this runs the actual training process
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.1,
  verbose = 1,
  callbacks = list(stopper, checker))

# load the best model from this training run
model = load_model_hdf5("best_model_0.h5")

# predict on test set for realistic evaluation
scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

# print out results
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')
