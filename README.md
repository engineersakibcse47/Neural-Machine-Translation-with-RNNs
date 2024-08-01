# Neural Machine Translation with RNNs(Seq2Seq) - Encoder Decoder

## Introduction
This project focuses on developing a neural machine translation (NMT) model using Recurrent Neural Networks (RNNs) to translate text from English to French. The model leverages various machine learning and natural language processing techniques.

## Data Preprocessing
```
text_dataset = tf.data.TextLineDataset("/content/dataset/fra.txt")

english_vectorize_layer=TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=ENGLISH_SEQUENCE_LENGTH
)
     
french_vectorize_layer=TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=FRENCH_SEQUENCE_LENGTH
)

```

## Neural Network Architecture
<img width="600" alt="image" src="https://github.com/user-attachments/assets/6b5e0015-706d-437a-8692-942532af8c53">

```
NUM_UNITS=256
     
### ENCODER
input = Input(shape=(ENGLISH_SEQUENCE_LENGTH,), dtype="int64", name="input_1")
x=Embedding(VOCAB_SIZE, EMBEDDING_DIM, )(input)
encoded_input=Bidirectional(GRU(NUM_UNITS), )(x)

### DECODER
shifted_target=Input(shape=(FRENCH_SEQUENCE_LENGTH,), dtype="int64", name="input_2")
x=Embedding(VOCAB_SIZE,EMBEDDING_DIM,)(shifted_target)
x = GRU(NUM_UNITS*2, return_sequences=True)(x, initial_state=encoded_input)

### OUTPUT
x = Dropout(0.5)(x)
target=Dense(VOCAB_SIZE,activation="softmax")(x)
seq2seq_gru=Model([input,shifted_target],target)
```

## Training the Model
```
seq2seq_gru.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(5e-4),
    metrics=["accuracy"])
    #metrics=[BLEU()],
    #run_eagerly=True)
```
### Loss Functions:
- SparseCategoricalCrossentropy

### Metrics:
- Accuracy
- BLEU Score

### Optimizer:
- Adam

## Results and Evaluation
The model's performance is evaluated using various metrics such as accuracy. Visualizations of the training process and results are provided using TensorBoard and other plotting libraries.
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/14b616fc-bf92-41d4-abec-70180188d08f">

## Conclusion
This project demonstrates the effectiveness of RNN-based architectures for neural machine translation. Further improvements can be made by experimenting with different model architectures, hyperparameters, and data augmentation techniques.

## Future Work
- Explore transformer-based models which have shown superior performance in recent years.
- Incorporate more diverse datasets to improve the robustness and generalization of the model.
- Optimize the training process for faster convergence and better resource utilization.
