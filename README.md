### FINAL PROJECT IDEA ###

For my final project, I made a voice authentication software. Unlike traditional speaker 
recognition where the model classifies different people, I built a security software that can validate or
deny certain users of certain voice using recurrent neural networks and lstm. 


```python
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_users, activation='sigmoid'))  # One output neuron per user
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
