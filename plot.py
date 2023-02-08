import matplotlib.pyplot as plt

x = [64, 96, 128, 160, 192, 224, 256]
y = [46.44, 54.42, 54.8, 55.5, 56.0, 56.7, 57.0]
plt.plot(x, y, "ro-", label="Baseline")
plt.xticks(x)
# plt.ylim(10, 60)
# plt.xlim(0, 30)

x = [64, 96, 128, 160, 192, 224, 256]
y = [46.48, 54.56, 55.24, 56.2, 57.7, 58.59, 59.98]
plt.plot(x, y, "bo-", label="Baseline + Ours")
plt.xticks(x)

# plt.tight_layout()
plt.xlabel("Spatial resolution")
plt.ylabel("mAP")
plt.legend(loc="best")

plt.savefig("plot.png", dpi=400)
plt.close()