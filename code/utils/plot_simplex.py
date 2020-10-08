

def plot_simplex(logits):
	import seaborn as sn
	import matplotlib.pyplot as plt
	import numpy as np
	from utils.simplex_plot_function import draw_pdf_contours, Dirichlet

	font = {'family': 'serif',
	            'color':  'black',
	            'weight': 'normal',
	            'size': 16,
	            }
	plt.style.use('seaborn-white')
	plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
	
	if len(logits[0,:]) == 3:
		for i in range(0,6):
			plt.subplot(2, 3, i+1)
			plt.title("logits: " + str(np.around(logits[i,:],decimals =1)) ,
			        fontsize=18, ha='center')
			plot_logits = logits[i,:]
			draw_pdf_contours(Dirichlet(plot_logits))
		
		

	
