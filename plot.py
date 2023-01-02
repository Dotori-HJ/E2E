import matplotlib.pyplot as plt

x = [5.61, 6.13, 28.0]
y = [54.45, 55.56, 57.83]
plt.plot(x, y, "r*-", label="224x224")
x = [1.80, 1.91, 7.11]
y = [47.64, 49.77, 52.62]
plt.plot(x, y, "b*-", label="96x96")

plt.xlabel("Memory (GB)")
plt.ylabel("mAP")
plt.legend(loc="best")

plt.savefig("plot.png")
plt.close()