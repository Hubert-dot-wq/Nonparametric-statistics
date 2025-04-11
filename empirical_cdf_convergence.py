import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.special import gamma
import matplotlib
from scipy.integrate import quad
matplotlib.use("TkAgg")
matplotlib.is_interactive()
matplotlib.get_backend()

def norm_pdf(x, mean, scale):
    return 1 / (np.sqrt(2*np.pi)*scale) * np.exp(-0.5*(x-mean/scale)**2)

size = 100
norm.cdf(np.linspace(-5, 5, 100))
def compare_cdf_ecdf(sizes):
    fig, axes = plt.subplots(1, 1, figsize=(8,8))
    axes.plot(np.linspace(-4, 4, 100), norm.cdf(np.linspace(-5, 5, 100)), label='True', color='red')

    res_dict = {}
    for size in sizes:
        X = norm.rvs(size=size)
        X_sorted = np.sort(X)
        X_emp_cdf = np.arange(len(X_sorted))/len(X_sorted)
        X_true_cdf = norm.cdf(X_sorted)

        axes.step(X_sorted, X_emp_cdf)

        # How close is the Empirical CDF to True CDF?
        # 0. error
        errors = (X_true_cdf - X_emp_cdf)**2
        errors_abs = np.abs(X_true_cdf - X_emp_cdf)
        # Errors spike at the shoulders. Why do errors around the mean are small? Why the errors at the tails are small?
        print("Error analysis, size=", size, "\n", 50*"=", "\n", 50*"=")
        # 1. MAE
        MAE = errors_abs.mean()
        print('MAE: ', MAE)
        # 2. MSE
        MSE = np.mean(errors)
        print("MSE: ", MSE)
        # Both MAE and MSE are small but those put equal weight on all observations.
        # How can we weight errors so that they appropriately represent all parts of the distribution?
        # 3. Kolmogorov-Smirnov statistic:
        KS_stat = (X_true_cdf-X_emp_cdf).max()
        print("Kolmogorov-Smirnov statistic: ", KS_stat)

        # 4. Cramer von Mises distance - density weighted error
        dx = np.diff(X_sorted)
        dx = np.append(dx, dx[-1])
        CM_distance = np.sum(
            (X_true_cdf-X_emp_cdf)**2*norm_pdf(X_sorted, 0, 1)*dx
        )
        print("CM_distance is: ", CM_distance)

        # 5. Anderson-Darling distance - puts more weights on tails

        AD_distance = np.sum(
            (X_true_cdf-X_emp_cdf)**2/np.clip((X_emp_cdf*(1-X_emp_cdf)), 1e-8, None)*norm_pdf(X_sorted, 0, 1)*dx
        )
        print("AD distance: ", AD_distance)
        # 6. Cramer von Mises distance with observation weighted summands
        X_sorted_tail = X_sorted[(np.percentile(X_sorted, 0.05) > X_sorted)| (X_sorted > np.quantile(X_sorted, 0.95))]
        X_true_cdf_tail = X_true_cdf[(np.percentile(X_true_cdf, 0.05) > X_true_cdf) | (X_true_cdf > np.quantile(X_true_cdf, 0.95))]
        X_emp_cdf_tail = X_emp_cdf[(np.percentile(X_emp_cdf, 0.05) > X_emp_cdf)| (X_emp_cdf> np.quantile(X_emp_cdf, 0.95))]
        tail_distance = np.abs(X_true_cdf_tail-X_emp_cdf_tail).mean()
        print(f"tail_distance (MAE )is: {tail_distance}")

        print("\n", 50 * "=")
        # Update res_dict
        res_dict[size] = [MAE, MSE, KS_stat, CM_distance, AD_distance, tail_distance]
    legend = ["true cdf"]
    for s in sizes:
        legend.append(s)
    plt.legend(legend)

    return axes, res_dict



axes, results = compare_cdf_ecdf([50, 100, 200, 1000])

def t_pdf(x, df):
    return gamma(0.5*(df+1)) / (np.sqrt(np.pi * df) * gamma(df/2)) * (1 + x**2/df)**(-0.5*(df+1))
