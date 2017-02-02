from matplotlib import pyplot as plt
import numpy as np

def draw_pdf(l, city):
	hist, bins = np.histogram(l, bins=(np.max(l) - np.min(l)) / 2)
	# hist, bins = np.histogram(lengths, bins=np.logspace(1, math.log10(np.max(lengths)),(np.max(lengths) - np.min(lengths)) / 50 ))
	# hist, bins = np.histogram(lengths, bins=np.logspace(1, math.log10(np.max(lengths)),200 ))
	hist = hist / float(np.sum(hist))
	# colors = iter(cm.rainbow(np.linspace(0, 1, 6)))
	# plt.scatter(bins[:-1], hist, color=next(colors))
	print 'max len:', max(l)
	fig = plt.figure(figsize=(9, 6), facecolor='black')
	ax = plt.subplot(1, 1, 1)

	plt.scatter(bins[:-1], hist, color='yellow')
	print hist[:20]
	print bins[:20]
	# plt.scatter(ls, freq, marker='o', color='red', s=0.5)
	plt.xlabel('Number of GPS points', fontsize=20)
	plt.ylabel('p(f)', fontsize=20)
	#plt.title(city, fontsize=20, color='white')
	ax.set_xlim(xmin=0, xmax=100)
	ax.set_ylim(ymin=0)#, ymax=0.2)
	#ax.set_xticks(range(0, 200, 10))
	ax.spines['bottom'].set_color('white')
	ax.xaxis.label.set_color('white')
	ax.spines['left'].set_color('white')
	ax.yaxis.label.set_color('white')

	ax.tick_params(axis='x', colors='white')
	ax.tick_params(axis='y', colors='white')

	# plt.show()
	plt.savefig('figs/trip_len_dist.png', format='PNG', bbox_inches='tight',  facecolor=fig.get_facecolor(), transparent=True)


lens = []
with open('data/trip_len_distibution.txt') as f:
	for line in f:
		lens.append(int(line.strip().split(' ')[0]))

from collections import Counter
cnt = Counter(lens)
print cnt.most_common(20)
draw_pdf(lens, city='doha')

