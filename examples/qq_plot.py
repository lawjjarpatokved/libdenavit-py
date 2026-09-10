import matplotlib.pyplot as plt
import numpy as np

from libdenavit import qq_coordinates


# Generate sample data
rng = np.random.default_rng(42)
data = rng.normal(loc=50, scale=5, size=100)

# Calculate Q-Q plot coordinates
x_qq, y_qq = qq_coordinates(data)

# Plot the Q-Q coordinates
fig, ax = plt.subplots()
ax.scatter(x_qq, y_qq)

ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Sample quantiles")
ax.set_title("Normal Q-Q Plot")

plt.show()