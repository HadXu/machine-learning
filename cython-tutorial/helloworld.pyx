from __future__ import print_function
def fib(n):
	a,b = 0,1
	while b<n:
		print(b, end=' ')
		a,b = b, a + b
	print()