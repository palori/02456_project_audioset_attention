import numpy as np
import pandas as pd

def hola():
	return "A1", "B2"

def hola2():
	a = hola()
	print("a: ",a, " type: ", type(a))
	return a

def test_hola():
	a,b = hola()
	print("a: ",a, " type: ", type(a))
	print("b: ",b, " type: ", type(b))

	a,b  = hola2()
	print("a: ",a, " type: ", type(a))
	print("b: ",b, " type: ", type(b))

def delete_row():
	#arr = np.array([[[1,2,3,4], [5,6,7,8], [9,10,11,12]],  [[11,12,13,14], [15,16,17,18], [19,110,111,112]]])
	arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
	
	print("\narr = \n", arr)
	arr = np.delete(arr, 1, 0)
	print("\narr = \n", arr)


def test_pandas():
	df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),
		columns=['a', 'b', 'c', 'd', 'e'])
	print(df2)
	df2['in']= [0, 1, 12, 13, 14]
	print(df2.iloc[3])
	print(df2.iloc[3]['b'])
	print(df2.index)


if __name__ == '__main__':
	#test_hola()
	#delete_row()
	test_pandas()



	


