import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("monitor.csv", skiprows=1)
# monitor.csv: r = reward, l = episode length
plt.plot(df["l"].cumsum(), df["r"].rolling(20).mean())
plt.xlabel("Episode")
plt.ylabel("Reward mean 20 episodes")
plt.title("TinyArena Training Curve")
plt.savefig("curve.png")
print("saved curve.png")
