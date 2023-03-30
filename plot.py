import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6.8))
plt.rc('font', size=17)        # 기본 폰트 크기
marker_size = 15

x = [64, 96, 128, 160, 192, 224, 256]
y1 = [46.44, 54.42, 54.8, 55.5, 56.0, 56.7, 57.0]
# plt.plot(x, y1, color="tab:green", linestyle="dashed", marker=".", markersize=12, label="Baseline", markerfacecolor='w', linewidth=2)
plt.plot(x, y1, color=plt.cm.Spectral(207), linestyle="dashed", marker=".", markersize=marker_size+1, label="Baseline", markerfacecolor='w', linewidth=2, markeredgewidth=3)
# plt.plot(x, y1, color=plt.cm.RdYlGn(10), linestyle="dashed", marker=".", markersize=12, label="Baseline", markerfacecolor='w', linewidth=2)
# plt.plot(x, y1, color="w", linestyle="", marker=".", markersize=4)
# plt.scatter(x, y1, color="tab:green", marker='ox')
plt.xticks(x)

x = [64, 96, 128, 160, 192, 224, 256]
y2 = [46.48, 54.56, 55.24, 56.2, 57.7, 58.8, 60.6]
# plt.plot(x, y2, color="tab:orange", linestyle="solid", marker="*", markersize=12, label="Baseline + Ours", markerfacecolor='w', linewidth=2)
plt.plot(x, y2, color=plt.cm.Spectral(17), linestyle="solid", marker="*", markersize=marker_size, label="Baseline + Ours", markerfacecolor='w', linewidth=2, markeredgewidth=2)
# plt.plot(x, y2, color=plt.cm.RdYlGn(200), linestyle="solid", marker="*", markersize=12, label="Baseline + Ours", markerfacecolor='w', linewidth=2)
# plt.plot(x, y2, color="w", linestyle="", marker="*", markersize=4)
plt.xticks(x)
# print(plt.cm.RdYlBu(200))

# plt.plot([200], [60], color="tab:green", linestyle="solid", marker=".", markersize=12)
# plt.plot([200], [60], color="w", linestyle="", marker=".", markersize=4)

# plt.plot([200], [50], color="tab:orange", linestyle="solid", marker="*", markersize=12)
# plt.plot([200], [50], color="w", linestyle="", marker="*", markersize=4)

# plt.annotate('', xy=(, 0), xytext=(5, 1), arrowprops=dict(arrowstyle='->'))
plt.fill_between(x, y1, y2, color='gray', alpha=0.15)

# plt.tight_layout()
plt.xlabel("Spatial resolution")
plt.ylabel("mAP")
plt.legend(loc="best")

plt.savefig("plot.png", dpi=400)
plt.close()