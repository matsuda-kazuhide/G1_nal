"""
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import linear_model, metrics, preprocessing, model_selection #機械学習用のライブラリを利用
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import emnist
import pickle
import matplotlib.pyplot as plt

train_data, train_label = emnist.extract_training_samples("letters")
test_data, test_label = emnist.extract_test_samples("letters")

#"""for origin
train_data = train_data.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2])
test_data = test_data.reshape(test_data.shape[0],test_data.shape[1]*test_data.shape[2])
#"""

train_label_parts = np.split(train_label,10)
#test_label_parts = np.split(test_label,10)

"""for dcf
train_data=np.load('train_dcf_v3.npy')
test_data=np.load('test_dcf_v3.npy')
"""




train_data_parts = np.array(np.split(train_data,10))

#test_data_parts = np.split(train_data,10)
#splitしたら戻り値はlist


"""
train_data_parts = np.split(train_data_parts,10)
test_data_parts = np.split(train_data_parts,10)
train_label_parts = np.split(train_label_parts,10)
test_label_parts = np.split(test_label_parts,10)

"""


def main():
	clf = SGDClassifier()
	result =[]
	for i in range(10):
		i+=1
		train_data_part = np.array(train_data_parts[0:i])
		train_label_part = np.array(train_label_parts[0:i])
		train_data_part = train_data_part.reshape(train_data_part.shape[0]*train_data_part.shape[1],train_data_part.shape[2])
		train_label_part = train_label_part.reshape(train_label_part.shape[0]*train_label_part.shape[1])
		clf.fit(train_data_part,train_label_part)
		test_predict = clf.predict(test_data)      
		ac = accuracy_score(test_label,test_predict)
		result.append(ac)
		#pickle.dump(clf,open('SGDpm.sav','wb'))
		print("{}で{}".format(i,ac))
	fig = plt.figure()
	plt.title('SGD_LearningProcess_WithOriginData')
	plt.xlabel('amount of data used')
	plt.ylabel('accuracy_score')
	x = [(i+1)/10 for i in range(10)]
	plt.plot(x,result)
	plt.xlim(0,1)
	plt.ylim(0,1)
	fig.savefig('SGD_lw_wdcv3.png')
#v3に書き換え予定
#main()にreshape書いてる
		
"""
clf.fit(train_data,train_label)
test_predict = clf.predict(test_data)
accuracy_score(test_label,test_predict)
"""
if __name__ == '__main__':
	main()
	
	


