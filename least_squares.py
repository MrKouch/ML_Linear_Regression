import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
data = sio.loadmat("lsdata.mat")
X, y = data["X"], data["Y"].ravel()
X_test, y_test = data["Xtest"], data["Ytest"].ravel()

# -----------------------------
# Models
# -----------------------------
def least_squares_w(Xm, ym):
    # min ||Xm w - ym||^2
    w, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    return w

def ridge_w(Xm, ym, lam):
    # min ||Xm w - ym||^2 + lam * ||w||^2
    # closed form: w = (Xm^T Xm + lam I)^(-1) Xm^T ym
    d = Xm.shape[1]
    A = Xm.T @ Xm + lam * np.eye(d)
    b = Xm.T @ ym
    return np.linalg.solve(A, b)

def average_squared_loss(X, y, w):
    return np.mean((X @ w - y) ** 2)

# -----------------------------
# (a) Least Squares vs m (your original part, fixed save/show order)
# -----------------------------
ms = np.arange(100, 501, 10)
rng = np.random.default_rng(0)
n = X.shape[0]

train_losses = []
test_losses = []

for m in ms:
    idx = rng.choice(n, size=m, replace=False)
    Xm, ym = X[idx], y[idx]

    w = least_squares_w(Xm, ym)

    # (keep your original behavior: loss on sampled train subset)
    train_losses.append(average_squared_loss(Xm, ym, w))
    test_losses.append(average_squared_loss(X_test, y_test, w))

plt.figure()
plt.plot(ms, test_losses)
plt.xlabel("m")
plt.ylabel("Average squared loss (test)")
plt.title("Least Squares: Test loss vs m")
plt.tight_layout()
plt.savefig("LS_test_loss.png", dpi=200)
plt.show()

plt.figure()
plt.plot(ms, train_losses)
plt.xlabel("m")
plt.ylabel("Average squared loss (train subset)")
plt.title("Least Squares: Train loss vs m")
plt.tight_layout()
plt.savefig("LS_train_loss.png", dpi=200)
plt.show()

# -----------------------------
# (b) Ridge Regression vs lambda, with horizontal LS baseline
# -----------------------------
lambdas = [0, 0.01, 0.02, 0.05, 0.1, 1, 10, 15]

def ridge_experiment(m, seed):
    rng_local = np.random.default_rng(seed)
    idx = rng_local.choice(n, size=m, replace=False)
    Xm, ym = X[idx], y[idx]

    # Least Squares baseline on SAME sampled data
    w_ls = least_squares_w(Xm, ym)
    ls_test_loss = average_squared_loss(X_test, y_test, w_ls)

    ridge_test_losses = []
    for lam in lambdas:
        if lam == 0:
            # exactly LS
            w = w_ls
        else:
            w = ridge_w(Xm, ym, lam)
        ridge_test_losses.append(average_squared_loss(X_test, y_test, w))

    return ls_test_loss, np.array(ridge_test_losses)

# (b-i, b-ii) m = 60
ls60, ridge60 = ridge_experiment(m=60, seed=1)

plt.figure()
plt.plot(lambdas, ridge60, marker="o")
plt.axhline(ls60, linestyle="--", label=f"LS baseline (m=60): {ls60:.4f}")
plt.xlabel("lambda (λ)")
plt.ylabel("Average squared loss (test)")
plt.title("Ridge Regression: Test loss vs λ (m=60)")
plt.legend()
plt.tight_layout()
plt.savefig("Ridge_test_loss_vs_lambda_m60.png", dpi=200)
plt.show()

# (b-iii, b-iv) m = 500
ls500, ridge500 = ridge_experiment(m=500, seed=2)

plt.figure()
plt.plot(lambdas, ridge500, marker="o")
plt.axhline(ls500, linestyle="--", label=f"LS baseline (m=500): {ls500:.4f}")
plt.xlabel("lambda (λ)")
plt.ylabel("Average squared loss (test)")
plt.title("Ridge Regression: Test loss vs λ (m=500)")
plt.legend()
plt.tight_layout()
plt.savefig("Ridge_test_loss_vs_lambda_m500.png", dpi=200)
plt.show()

# Optional: print the numbers so you can describe behavior in text
print("m=60:  LS test loss =", ls60)
print("m=60:  Ridge test losses by lambda =", dict(zip(lambdas, ridge60)))
print("m=500: LS test loss =", ls500)
print("m=500: Ridge test losses by lambda =", dict(zip(lambdas, ridge500)))
