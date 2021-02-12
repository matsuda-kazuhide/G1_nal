from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import emnist
import matplotlib.pyplot as plt
import pickle

train_data, train_label = emnist.extract_training_samples("letters")
test_data, test_label = emnist.extract_test_samples("letters")

"""
train_data = train_data.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2])
test_data = test_data.reshape(test_data.shape[0],test_data.shape[1]*test_data.shape[2])
"""

#train_data = np.load('train_dcf_v3.npy')
#test_data = np.load("test_dcf_v3.npy")

train_data_parts = np.split(train_data,10)
train_label_parts = np.split(train_label,10)


def main():
	knc = KNeighborsClassifier(n_neighbors=5)
	for i in range(10):
		i+=1
		train_data_part = np.array(train_data_parts[0:i])
		train_label_part = np.array(train_label_parts[0:i])
		train_data_part = train_data_part.reshape(train_data_part.shape[0]*train_data_part.shape[1],train_data_part.shape[2])
		train_label_part = train_label_part.reshape(train_label_part.shape[0]*train_label_part.shape[1])
		knc.fit(train_data_part, train_label_part)
		test_predict = knc.predict(test_data)
		ac = accuracy_score(test_label,test_predict)
		print("{}で{}".format(i,ac))
		#pickle.dump(clf,open('knn_r.sav','wb'))
	print('use dcf')

	

if __name__ == '__main__':
	main()