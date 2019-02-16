import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 2.5))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.rcParams['grid.linestyle'] = "-"
plt.grid(True)

# a = [1000, 3397, 1849, 2501, 2782, 1873, 1000]

# b = [1000, 708, 966, 1045, 1307, 836, 1000]
# offset_b = np.add(a, b)

# c = [1000, 3310, 1676, 2851, 3458, 1528, 1000]
# offset_c = np.add(c, offset_b)

# d = [1000, 2199, 1681, 2603, 2198, 1853, 1000]
# offset_d = np.add(d, offset_c)

# p1 = plt.bar([x - 0.3 for x in range(7)], a, 0.2, alpha=0.4, color='r', zorder=5)
# p2 = plt.bar([x - 0.1 for x in range(7)], b, 0.2, alpha=0.4, color='b', zorder=5)
# p3 = plt.bar([x + 0.1 for x in range(7)], c, 0.2, alpha=0.4, color='g', zorder=5)
# p4 = plt.bar([x + 0.3 for x in range(7)], d, 0.2, alpha=0.4, color='y', zorder=5)

# plt.ylabel('Episodes',)
# plt.xticks(range(7), ('40-TSP', '50-TSP', '60-TSP', '70-TSP', '80-TSP', '90-TSP', '100-TSP'))

a = [1000, 5000, 1000, 3000, 1000, 1000]

b = [1000, 3000, 1000, 1000, 5000, 1000]
offset_b = np.add(a, b)

c = [1000, 1000, 1000, 5000, 3000, 1000]
offset_c = np.add(c, offset_b)

d = [3000, 5000, 1000, 1000, 1000, 1000]
offset_d = np.add(d, offset_c)

p1 = plt.bar([x - 0.3 for x in range(6)], a, 0.2, alpha=0.4, color='r', zorder=5)
p2 = plt.bar([x - 0.1 for x in range(6)], b, 0.2, alpha=0.4, color='b', zorder=5)
p3 = plt.bar([x + 0.1 for x in range(6)], c, 0.2, alpha=0.4, color='g', zorder=5)
p4 = plt.bar([x + 0.3 for x in range(6)], d, 0.2, alpha=0.4, color='y', zorder=5)

plt.ylabel('Episodes',)
plt.xticks(range(6), ('40- to 50-TSP', '50 to 60-TSP', '60 to 70-TSP', '70 to 80-TSP', '80- to 90-TSP', '90- to 100-TSP'))

plt.legend((p1[0], p2[0], p3[0], p4[0]), ('SARSA(Table)', 'SARSA(Fourier)', 'Q-learning(Table)', 'Q-learning(Fourier)'), loc=1, ncol=2, fontsize='x-small')

plt.tight_layout(True)
plt.show()
