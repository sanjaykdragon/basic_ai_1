from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os

#credits to https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/ for the tutorial, this is just my basic take on it
#a lot of the code is from ^

dataset = loadtxt('C:\\Users\\Null\\Desktop\\python_garbage\\test.csv', delimiter=',')

start_index = 0
numbers_to_take = 2
output_index = 2

example_input = dataset[: , start_index : numbers_to_take]
example_output = dataset[: , output_index]

model = Sequential()
model.add(Dense(12, input_dim=numbers_to_take, activation='relu'))
model.add(Dense(8, activation='relu')) #no idea what this is even used for
model.add(Dense(1, activation='sigmoid')) #ranges from 0 to 1 - probability https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(example_input, example_output, epochs=150) #do the learning

predictions = model.predict_classes(example_input) #do the actual prediction

os.system('cls') #for some reason a load of random tensorflow garbage gets dumped, so im cleaning this garbage

size_of_inp = len(example_input)
right = 0
wrong = 0
for i in range(size_of_inp): 
	inp = example_input[i].tolist()
	pred = predictions[i]
	outp = example_output[i]
	print('%s => %d (expected %d)' % (inp, pred, outp))
	if pred != outp:
		wrong += 1
	else:
		right += 1

print("%s right %s wrong %s percent" % (right, wrong, (right / size_of_inp)))