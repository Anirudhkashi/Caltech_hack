import numpy as np

BALL_ANNOTATION_FILE = "dataset.txt"

file_names = []
POS = []
Y = []

all_out = []

with open(BALL_ANNOTATION_FILE, "r") as f:

	for line in f:

		temp_all = []
		p = []
		line = line.strip().split(",")
		f_name = line[1]

		temp_all.append(f_name)

		# Read ball positions
		x = line[4]
		y = line[5]

		p.append(x)
		p.append(y)

		direc = f_name.split("-")[0]
		annotation_name = f_name.split(".")[0] + "-annotations.txt"

		# Read the annotation - the position of players

		try:
			with open("annotations/" + annotation_name) as f_in:
				tmp = f_in.read().strip().split("\n")
				p = p + tmp

			temp_all.append(p)
		except:
			continue
		
		# Read target variables

		
		try:
			target_name = f_name.split(".")[0] + "-target.txt"
			with open("targets/" + direc + "/" + target_name) as f_in:
				tmp = f_in.read().strip().split("\n")
				temp_all.append(tmp)
		except:
			continue
			
		all_out.append(temp_all)


all_out = sorted(all_out, key= lambda x: (x[0].split("-")[0], int(x[0].split("frame")[1].split(".")[0])))

for ex in all_out:

	file_names.append(ex[0])
	POS.append(ex[1])
	Y.append(ex[2])

TRAIN_SPLIT = 0.8

examples = len(all_out)
num_splits = 5

splits = int(examples / num_splits)
split_tuple_index = [(0, splits), (splits, splits * 2), (splits * 2, splits * 3), (splits * 3, splits * 4), (splits * 4, )]

def getAnnotation(cross_validation=5):

	POS_LENGTH = 6
	TIME_STEPS = 15

	split_type = split_tuple_index[cross_validation-1]

	if cross_validation == 5:
		X_train = POS[:split_type[0]]
		Y_train = Y[:split_type[0]]
		X_test = POS[split_type[0]:]
		Y_test = Y[split_type[0]:]

	elif cross_validation == 1:
		X_train = POS[split_type[1]:]
		Y_train = Y[split_type[1]:]
		X_test = POS[:split_type[1]]
		Y_test = Y[:split_type[1]]

	else:
		X_train = POS[:split_type[0]] + POS[split_type[1]:]
		Y_train = Y[:split_type[0]] + Y[split_type[1]:]
		X_test = POS[split_type[0]: split_type[1]]
		Y_test = Y[split_type[0]: split_type[1]]


	total_train = int(len(X_train) / 15)
	total_test = int(len(X_test) / 15)
	X_train = X_train[:total_train * 15]
	Y_train = Y_train[:total_train * 15]
	X_test = X_test[: total_test * 15]
	Y_test = Y_test[: total_test * 15]

	X_train = np.array(X_train).reshape(total_train, TIME_STEPS, POS_LENGTH)
	Y_train = np.array(Y_train).reshape(total_train, TIME_STEPS, 1)

	Y_train = np.swapaxes(Y_train, 0, 1)

	X_test = np.array(X_test).reshape(total_test, TIME_STEPS, POS_LENGTH)
	Y_test = np.array(Y_test).reshape(total_test, TIME_STEPS, 1)

	Y_test = np.swapaxes(Y_test, 0, 1)

	return X_train, Y_train, X_test, Y_test


def getGenerativeSequence():

	POS_LENGTH = 6
	TIME_STEPS = 15

	X_train = POS[:-20]
	Y_train = POS[1:-19]
	X_test = POS[-20:-1]
	Y_test = POS[-19:]

	total_train = int(len(X_train) / 15)
	total_test = int(len(X_test) / 15)
	X_train = X_train[:total_train * 15]
	Y_train = Y_train[:total_train * 15]
	X_test = X_test[: total_test * 15]
	Y_test = Y_test[: total_test * 15]

	X_train = np.array(X_train).reshape(total_train, TIME_STEPS, POS_LENGTH)
	Y_train = np.array(Y_train).reshape(total_train, TIME_STEPS, POS_LENGTH)

	X_test = np.array(X_test).reshape(total_test, TIME_STEPS, POS_LENGTH)
	Y_test = np.array(Y_test).reshape(total_test, TIME_STEPS, POS_LENGTH)

	return X_train, Y_train, X_test, Y_test

def getPredictingData():

	POS_LENGTH = 6
	TIME_STEPS = 15

	X_predict = [POS[-15:]]
	X_predict = np.array(X_predict).reshape(1, TIME_STEPS, POS_LENGTH)
	return X_predict, file_names[-15:]


# X_train, Y_train, X_test, Y_test = getAnnotation(cross_validation=5)
# print X_train.shape
# print Y_train.shape
# print X_test.shape
# print Y_test.shape