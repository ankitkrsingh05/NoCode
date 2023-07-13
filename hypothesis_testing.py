from modules.packages import np,plt,stats,sm
from base import NormalityTestsBase


class NormalityAnalyticalTests(NormalityTestsBase):
    def perform_test(self):
        self.shapiro_stat, self.shapiro_p = stats.shapiro(self.data)
        self.anderson_result = stats.anderson(self.data)
        self.kstest_stat, self.kstest_p = stats.kstest(self.data, 'norm')
    
    def display_results(self):
        tests = [
            ("Shapiro-Wilk test:", self.shapiro_stat, self.shapiro_p),
            ("Anderson-Darling test:", self.anderson_result.statistic, self.anderson_result.significance_level),
            ("Kolmogorov-Smirnov test:", self.kstest_stat, self.kstest_p)
        ]
        
        for test_name, test_stat, test_p in tests:
            print(test_name)
            print("Test Statistic:", test_stat)
            print("p-value:", test_p)
            print()

class NormalityVisualTests(NormalityTestsBase):
    def plot_histogram_with_bell_curve(self):
        plt.hist(self.data, bins='auto', density=True, alpha=0.7)
        mu, sigma = stats.norm.fit(self.data)
        x = np.linspace(np.min(self.data), np.max(self.data), 100)
        y = stats.norm.pdf(x, mu, sigma)
        
        plt.plot(x, y, 'r-', label='Normal Distribution')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()

    def plot_qq_plot(self):
        sm.qqplot(self.data, line='s')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.show()

    def perform_test(self):
        pass
    
    def display_results(self):
        pass

# Sample data
data = np.random.normal(loc=0, scale=1, size=1000)

# Perform analytical normality tests
analytical_tests = NormalityAnalyticalTests(data)
analytical_tests.perform_test()
analytical_tests.display_results()

# Perform visual normality tests
visual_tests = NormalityVisualTests(data)
visual_tests.plot_histogram_with_bell_curve()
visual_tests.plot_qq_plot()