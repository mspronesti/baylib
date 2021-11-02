import baylib

print('This is an example of usage of python-baylib')
b = baylib.bayesian_net()

# add 10 variables
for i in range(5):
    b.add_variable()

"""
Let's build this network

1 --> 0 <--4 
^     ^    ^  
|     |    | 
2     ---- 3
"""

b.add_dependency(1, 0)
b.add_dependency(4, 0)
b.add_dependency(3, 4)
b.add_dependency(2, 1)
b.add_dependency(3, 0)

print(b.has_dependency(1, 0))  # true
print(b.has_dependency(0, 3))  # false
print(b.is_root(0))  # false
print(b.is_root(3))  # true
