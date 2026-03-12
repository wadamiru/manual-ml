from data import make_polynomial_data
from model import fit_ols, predict

def main():
    # config
    nsamples = 100
    degree = 3
    
    X, y = make_polynomial_data(nsamples, degree)

    theta = fit_ols(X, y)
    ypred = predict(X, theta)

    print("theta: ", theta)

if __name__ == "__main__":
    main()