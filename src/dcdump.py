import pandas as pd
import numpy as np
import emnist

train_data, train_label = emnist.extract_training_samples("letters")
test_data, test_label = emnist.extract_test_samples("letters")


def cknp(x,y):   #枠の外か内かの判定
	if x < 1 or y < 1:
		return False
	elif x > 26 or y > 26:
		return False
	else:
		return True
"""
def ckc(target,p,x,y):  #白か黒かの判定
	c = target[p][x][y]
	if c < 200:
		False
	else:
		True
"""
def ckc(target,p,x,y):  #白か黒かの判定
		if cknp(x,y)==True:
			c = target[p][x][y]
			if c < 200:
				return False
			else:
				return True
				
def search_len(target,p,x,y,looking_derection): #looking_derection は長さ２のリスト
	a=1
	while ckc(target,p,x-a*looking_derection[0],y-a*looking_derection[1])==True:
		a+=1
	return a-1
	
def lookup(target,p,x,y): #
	up_down,rup_ldown,right_left,lup_rdown = 0,0,0,0
	for looking_derection in [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]:
		lenght = search_len(target,p,x,y,looking_derection)
		if looking_derection[0]==0:
			up_down+=lenght
		elif looking_derection[1]==0:
			right_left+=lenght
		elif looking_derection[0]*looking_derection[1] < 0:
			rup_ldown+=lenght
		elif looking_derection[0]*looking_derection[1] > 0:
			lup_rdown+=lenght
		else:
			print('なんかミスってるよ(方向線探索)')
	return up_down,rup_ldown,right_left,lup_rdown
		
def ddc(up_down,rup_ldown,right_left,lup_rdown): #dc特徴量dump
	if up_down==0:
		up_down_dcf=0
	else:
		up_down_dcf = up_down / ((up_down)**2+(rup_ldown)**2+(right_left)**2+(lup_rdown)**2)**(1/2)
	if rup_ldown==0:
		rup_ldown_dcf=0
	else:
		rup_ldown_dcf = rup_ldown / ((up_down)**2+(rup_ldown)**2+(right_left)**2+(lup_rdown)**2)**(1/2)
	if right_left==0:
		right_left_dcf=0
	else:
		right_left_dcf = right_left / ((up_down)**2+(rup_ldown)**2+(right_left)**2+(lup_rdown)**2)**(1/2)
	if lup_rdown==0:
		lup_rdown_dcf=0
	else:
		lup_rdown_dcf = lup_rdown / ((up_down)**2+(rup_ldown)**2+(right_left)**2+(lup_rdown)**2)**(1/2)
	return up_down_dcf,rup_ldown_dcf,right_left_dcf,lup_rdown_dcf

def search(target,p,x,y):   #
	if ckc(target,p,x,y) == False:
		up_down_dcf,rup_ldown_dcf,right_left_dcf,lup_rdown_dcf = 0,0,0,0
	else:
		up_down,rup_ldown,right_left,lup_rdown = lookup(target,p,x,y)
		up_down_dcf,rup_ldown_dcf,right_left_dcf,lup_rdown_dcf = ddc(up_down,rup_ldown,right_left,lup_rdown)
	return up_down_dcf,rup_ldown_dcf,right_left_dcf,lup_rdown_dcf
		
"""
def looku(p,x,y):
	x,y = x,y
	if ckc(x,y-1):
		if x_train[p][x][y-1]>200
	else:
	
	
"""



def main():
	tn =0
	for target in [train_data,test_data]:
		iterate = target.shape[0]
		#iterate = 10
		f_all_list = np.zeros([iterate,26*26*4])
		for i in range(iterate):
			f_p_list = np.zeros([26,26*4])
			for y in range(26):	
				for x in range(26):
					up_down_dcf,rup_ldown_dcf,right_left_dcf,lup_rdown_dcf = search(target,i,x,y)
					f_p_list[y][x*4] = up_down_dcf
					f_p_list[y][x*4+1] = rup_ldown_dcf
					f_p_list[y][x*4+2] = right_left_dcf
					f_p_list[y][x*4+3] = lup_rdown_dcf
			f_p_list = f_p_list.reshape([26*26*4])
			f_all_list[i] = f_p_list
		#f_all_list = f_all_list.reshape([iterate,26*26*4]) 
		if tn == 0:
			np.save('/Users/e185708/downloads/ml/train_dcf',f_all_list)
			print(f_all_list.shape)
			tn+=1
		else:
			np.save('/Users/e185708/downloads/ml/test_dcf',f_all_list)
			print(f_all_list.shape)
			print('suc')

if __name__ == '__main__':
	main()