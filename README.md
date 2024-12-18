### FINAL PROJECT IDEA ###

For my final project, I made a voice authentication software. Unlike traditional speaker 
recognition where the model classifies different people, I built a security software that can validate or
deny certain users of certain voice using recurrent neural networks and lstm. 

The rnn intakes the mfcc (mel frequency ceptral coefficients) and interprets it through a time series. Mfcc's are simply put, 
notable features of an audio file. Since the human voice is complex, mfccs simply abbreviate the special features for everyone's voice. 
The lstm rnn helps retain information over time, getting rid of information and keeping it according to 
how important it is considered. 

My second approach is using the fourier transformation. Since a voice wave is essentially a collection of
different sinusoidal waves. What the other model does essentially is factor out those waves and then
assess how unique they are. Thankfully, there is a fourier transformation functionality in numpy. 

```python
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_users, activation='sigmoid'))  # One output neuron per user
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
fft_signal = np.fft.fft(audio_signal)
magnitude = np.abs(fft_signal) 
frequencies = np.fft.fftfreq(len(magnitude), d=1/sampling_rate) 
```
### How to Run ###
After forking the repository, navigate to the sound_test.ipynb file. Record your voice for five minutes, and then paste the path to the file to the function and run it. 
It should show the predictions and the probability vector. 


I failed to use markov chains. They are so difficult for some reason. 
