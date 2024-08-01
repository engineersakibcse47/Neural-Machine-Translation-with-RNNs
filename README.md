# Neural Machine Translation with RNNs(Seq2Seq) - Encoder Decoder

## Introduction
This project focuses on developing a neural machine translation (NMT) model using Recurrent Neural Networks (RNNs) to translate text from English to French. The model leverages various machine learning and natural language processing techniques.

## Data Extraction and Preprocessing
1. **Extract the downloaded zip file** to obtain the sentence pairs.
2. **Preprocess the data** to prepare it for model training.

## Neural Network Architecture
- **Embedding Layer**: Converts input tokens into dense vectors of fixed size.
- **Recurrent Layers (RNN/GRU/LSTM)**: Captures the sequential nature of the text data.
- **Dense Layers**: Used for generating the final translation output.
- **Other Layers**: Batch Normalization, Dropout for regularization.

## Training the Model
### Loss Functions:
- SparseCategoricalCrossentropy

### Metrics:
- Accuracy
- BLEU Score

### Optimizer:
- Adam

## Results and Evaluation
The model's performance is evaluated using various metrics such as accuracy. Visualizations of the training process and results are provided using TensorBoard and other plotting libraries.

## Conclusion
This project demonstrates the effectiveness of RNN-based architectures for neural machine translation. Further improvements can be made by experimenting with different model architectures, hyperparameters, and data augmentation techniques.

## Future Work
- Explore transformer-based models which have shown superior performance in recent years.
- Incorporate more diverse datasets to improve the robustness and generalization of the model.
- Optimize the training process for faster convergence and better resource utilization.
