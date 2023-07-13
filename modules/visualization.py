from packages import plt,stats,sns

def plot_qq_plot(data, title):
    plt.figure(figsize=(6, 4))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.show()

def plot_distribution(data, title):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def add_bell_curve(data, column):
    x = np.linspace(data[column].min(), data[column].max(), 100)
    y = stats.norm.pdf(x, data[column].mean(), data[column].std())
    plt.plot(x, y, color='red', label='Normal Distribution')
    plt.legend()

plot_qq_plot([[1,2,3]],'abc')