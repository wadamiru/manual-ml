from data import make_data
from model import LinearRegression
from train import train

# True parameters
w = 3.0
b = 2.0

x, y = make_data(w, b, nsamples=100, noise=False)

model = LinearRegression()
train(model, x, y, lr=0.1, epochs=1000, verbose=True)

print(f"True w: {w} | Learned w: {model.w:.4f}")
print(f"True b: {b} | Learned b: {model.b:.4f}")