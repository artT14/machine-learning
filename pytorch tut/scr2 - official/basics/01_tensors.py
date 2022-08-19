import torch
import numpy as np

# CREATE A TENSOR
data = [[69,420],[33,55]]
x_data = torch.tensor(data)
print("CREATING A NEW TENSOR:\n", x_data)

# CREATE A TENSOR FROM NUMPY ARRAYS
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("CREATING A TENSOR FROM NUMPY ARRAYS:\n", x_np)

# CREATE A SIMILIAR TENSOR TO ANOTHER, ONES
x_ones = torch.ones_like(x_data)
print("CREATING A TENSOR SIMILIAR TO ANOTHER, ONES:\n", x_ones)

# CREATE A SIMILIAR TENSOR TO ANOTHER, RANDOM
x_rand = torch.rand_like(x_data, dtype=torch.float)
print("CREATING A TENSOR SIMILIAR TO ANOTHER, RANDOM:\n",x_rand)

# CREATE A TENSOR BASED ON SHAPE
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(
	"RAND TENSOR:",rand_tensor,
	"ONES TENSOR:",ones_tensor,
	"ZEROS TENSOR:",zeros_tensor,
	sep='\n')

# ATTRIBUTES OF A TENSOR
tensor = torch.rand(3,4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device of tensor: {tensor.device}')

# MOVING TENSOR TO GPU
if torch.cuda.is_available():
	tensor = tensor.to('cuda')
	print("Moving tensor to GPU")
	print(f'Device of tensor: {tensor.device}')
	tensor = tensor.to('cpu')
	print("Moving tensor to CPU")
	print(f'Device of tensor: {tensor.device}')


# INDEXING & SLICING
tensor = torch.rand(4,4)
print(f'COMPLETE TENSOR: {tensor}')
print('First Row:', tensor[0])
print('Last Row:', tensor[-1])
print('First Column:', tensor[:,0])
print('Last Column:', tensor[...,-1])


# JOIN TENSORS
t1 = torch.cat([tensor,tensor,tensor], dim=1)
t2 = torch.cat([tensor,tensor,tensor], dim=0)
print(t1)
print(t2)

# TRANSPOSE TENSOR
tensor_tsps = tensor.T
print(tensor)
print(tensor_tsps)

# MATRIX MULTIPLY TENSORS
y1 = tensor @ tensor_tsps
y2 = tensor.matmul(tensor_tsps)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor_tsps, out=y3)
print(y1)
print(y2)
print(y3)

# ELEMENT-WISE MULTIPLY TENSORS
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)
print(z1)
print(z2)
print(z3)

# SINGLE ELEMENT TENSOR
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# IN PLACE OPERATIONS
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# BRIDGE W/ NUMPY
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')

t.add_(1)
print(f't: {t}')
print(f'n: {n}')
# NOTE: changes in torch will reflect in np & vice versa
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")