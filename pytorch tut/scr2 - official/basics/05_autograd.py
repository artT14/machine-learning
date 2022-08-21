# LINK: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
# AUTOGRAD IS USED FOR BACK PROPOGATION
# WEIGHTS OF MODEL ARE ADJUSTED BASED ON THE GRADIENT OF LOSS FUNCTION
import torch

x = torch.ones(5) # input
y = torch.zeros(3) # expected output
w = torch.randn(5,3,requires_grad=True) # paramater
# NOTE: can also do w.requires_grad_(True) later on
b = torch.randn(3,requires_grad=True) # paramater
z = torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# COMPUTING DERIVATIVES OF LOSS FUNCTION TO OPTIMIZED MODEL
loss.backward()
print(w.grad)
print(b.grad)

# DISABLE GRADIENT TRACKING
z = torch.matmul(x, w)+b
print(z.requires_grad)
    #as opposed to 
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
# NOTE: can also use z_det = z.detach() & use z_det to apply input to model
# WHY DETACH?
#   - Freeze Parameters to finetune the network
#   - Speed Up computation for forward pass only computations

# JACOBIAN PRODUCT
# NOTE: if output is a scalar, we use a loss function, if it is a tensor, we use the Jacobian Product
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")