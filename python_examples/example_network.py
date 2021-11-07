import baylib

print('This is an example of usage of python-baylib')
b = baylib.bayesian_net()

# add 10 variables
for i in range(6):
    b.add_variable()

"""
Let's build this network

1 --> 0 <--4   
^     ^    ^    
|     |    |   
2     ---- 3 ---> 5
"""

b.add_dependency(1, 0)
b.add_dependency(4, 0)
b.add_dependency(3, 4)
b.add_dependency(2, 1)
b.add_dependency(3, 0)
b.add_dependency(3, 5)

print(b.has_dependency(1, 0))  # true
print(b.has_dependency(0, 3))  # false
print(b.is_root(0))  # false
print(b.is_root(3))  # true

b.remove_dependency(3, 0)
print("3 and 0 have a dependency: ", b.has_dependency(3, 0))  # false

print(b.children_of(3))  # [4, 5]
print(b.parents_of(0))  # [1, 4]


g = baylib.gibbs_sampling(b, 10000, 1, 0)
out = g.make_inference()
