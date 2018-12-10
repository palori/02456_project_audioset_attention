def hola():
	return "A1", "B2"

def hola2():
	a = hola()
	print("a: ",a, " type: ", type(a))
	return a

if __name__ == '__main__':
	a,b = hola()
	print("a: ",a, " type: ", type(a))
	print("b: ",b, " type: ", type(b))

	a,b  = hola2()
	print("a: ",a, " type: ", type(a))
	print("b: ",b, " type: ", type(b))


	


