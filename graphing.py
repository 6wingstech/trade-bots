import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import numpy as np
import os
import time


def oscillator_graph(df, a, b, c=None, d=None, e=None, f=None, g=None):
	plotting = [a, b]
	if c != None:
		plotting.append(c)
	if d != None:
		plotting.append(d)
	if e != None:
		plotting.append(e)
	if f != None:
		plotting.append(f)
	if g != None:
		plotting.append(g)

	plots = df[plotting].plot(subplots=True, figsize=(10, 10))
	plt.show()

def overlay_graph(df, a, b, c=None, d=None, e=None, f=None, g=None, h=None):
	plotting = [a, b]
	if c != None:
		plotting.append(c)
	if d != None:
		plotting.append(d)
	if e != None:
		plotting.append(e)
	if f != None:
		plotting.append(f)
	if g != None:
		plotting.append(g)
	if h != None:
		plotting.append(h)

	plots = df[plotting].plot(subplots=False, figsize=(10, 4))
	plt.show()

def save_graph(df, name, triggers, j_algo, sell_list, a, b, c=None, d=None, e=None, f=None, g=None, h=None):
	plotting = [a, b]
	if c != None:
		plotting.append(c)
	if d != None:
		plotting.append(d)
	if e != None:
		plotting.append(e)
	if f != None:
		plotting.append(f)
	if g != None:
		plotting.append(g)
	if h != None:
		plotting.append(h)
	plots = df[plotting].plot(subplots=False, figsize=(20, 9))
	if triggers:
		for i in triggers:
			length = i[0]
			price = i[1]
			#message = 'V:' + str(i[2]) + ' Stdev:' + str(i[3])
			message = 'BUY (BP) ' + str(i[2])
			plt.text(length, price, message,
				horizontalalignment='center',
				verticalalignment='center',
				color='green',
				fontsize=14)

	if j_algo:
		for i in j_algo:
			length = i[0]
			price = i[1]
			message = 'BUY (J) ' + str(i[2])
			plt.text(length, price, message,
				horizontalalignment='center',
				verticalalignment='center',
				color='green',
				fontsize=14)

	if sell_list:
		for i in sell_list:
			length = i[0]
			price = i[1]
			if i[5] == 'Sell_Spike':
				message = 'SELL (SPIKE)'
			elif i[5] == 'Sell_Low_Vol':
				message = 'SELL (LV)'
			elif i[5] == 'Sell_4_Above':
				message = 'SELL (4)'
			elif i[5] == 'Sell_Downtrend':
				message = 'SELL (DT)'
			plt.text(length, price, message,
				horizontalalignment='center',
				verticalalignment='center',
				color='red',
				fontsize=14)


	plt.savefig(name)
	plt.close()

def save_graph_mm(df, name, buy_list, sell_list, a, b=None, c=None, d=None, e=None, f=None, g=None, h=None):
	plotting = [a]
	if b != None:
		plotting.append(b)
	if c != None:
		plotting.append(c)
	if d != None:
		plotting.append(d)
	if e != None:
		plotting.append(e)
	if f != None:
		plotting.append(f)
	if g != None:
		plotting.append(g)
	if h != None:
		plotting.append(h)
	plots = df[plotting].plot(subplots=False, figsize=(20, 9))
	if buy_list:
		for i in buy_list:
			length = i[0]
			price = i[1]
			#message = 'V:' + str(i[2]) + ' Stdev:' + str(i[3])
			message = 'BUY'
			plt.text(length, price, message,
				horizontalalignment='center',
				verticalalignment='center',
				color='green',
				fontsize=9)

	if sell_list:
		for i in sell_list:
			length = i[0]
			price = i[1]
			message = 'SELL'
			plt.text(length, price, message,
				horizontalalignment='center',
				verticalalignment='center',
				color='red',
				fontsize=9)


	plt.savefig(name)
	plt.close()

def scatter(df, a, b):	
	plt.scatter(df[a], df[b])
	plt.xlabel(a)
	plt.ylabel(b)
	plt.show()

