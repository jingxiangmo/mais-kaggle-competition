# MAIS 202 Winter 2023 Fashion MNIST Challenge - Jingxiang Mo

### Summary
> I implemented a Keras Sequential model, achieved an accuracy of 87.460% by tuning hyperparameters and dropout rates, and addressed overfitting with regularization techniques.

### Implementation of the Model

- Libraries: numpy, pandas, and Keras are used for loading and processing the dataset
- Keras Sequential model with the following architecture:
    - Input layer (28x28)
    - Reshape layer (28x28x1)
    - Conv2D layer (64 filters, 3x3 kernel, ReLU activation)
    - MaxPooling2D layer (2x2 pool size)
    - Dropout layer (0.25 dropout rate)
    - Conv2D layer (32 filters, 3x3 kernel, ReLU activation)
    - MaxPooling2D layer (2x2 pool size)
    - Dropout layer (0.25 dropout rate)
    - Flatten layer
    - Dense layer (128 units, ReLU activation, L1 and L2 regularization)
    - Dropout layer (0.5 dropout rate)
    - Output Dense layer (10 units, softmax activation)
- Trained for 100 epochs with a batch size of 128 and 10% validation split

### Results

I achieved an accuracy of 87.460% by tuning hyperparameters and dropout rate.

Hyperparameters I fine-tuned: 

- Epochs: The model was trained for 100 epochs. By experimenting with different numbers of epochs, I found that training for 100 epochs allowed the model to converge without overfitting.
- Batch size: I tested various batch sizes and settled on a batch size of 128.
- Dropout rates: To address the initial overfitting issue, I incorporated dropout layers in the model.
- Regularization: I also applied L1 and L2 regularization to reduce overfitting.

### Challenges

Overfitting was initially a problem, but I introduced dropout layers and L1/L2 regularization to enhanced the model's generalization ability.

A key challenge was the limited dataset size that limited the model accuracy, which could be addressed in the future by exploring techniques like data augmentation.

### Conclusion

- learned how to preprocess and prepare the data for training models.
- learned how to create a  convolutional neural network (CNN) using Keras and how to tune hyperparameters for improved performance
- learned the importance of adding regularization techniques, such as dropout and L1/L2 regularization, to reduce overfitting in the model.
- Learned how to read accuracy and loss graphs to fine tune the models.