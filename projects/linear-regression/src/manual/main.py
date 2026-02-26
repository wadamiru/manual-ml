from .data import make_data
from .model import LinearRegression
from .train import train

x, y = make_data(w=3.0, b=2.0, nsamples=100, noise=True)

model = LinearRegression()
train(model, x, y, lr=0.1, epochs=1000, verbose=True)

print("Learned w:", model.w)
print("Learned b:", model.b)