import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import openpyxl
import pandas as pd

plt.style.use('ggplot')

def freq_fundamental(t, i):
	zero_crossings = np.where(np.diff(np.sign(i)))[0]
	t_zc = t[zero_crossings]
	dt_fund = np.zeros(int(len(t_zc)))
	m = 0
	for k in range(0, len(t_zc) - 10):
		for cont in range(1, 10, 1):
			dt_zc = t_zc[k + cont] - t_zc[k]
			if 8.5e-3 > dt_zc > 8.25e-3:
				dt_fund[m] = 2 * dt_zc
				m = m + 1
				break

	freq_fund_temp = (1 / dt_fund[dt_fund != 0])
	freq_fund = np.round(np.average(freq_fund_temp),2)

	tlim1 = 0.999*dt_fund[0]
	tlim2 = 1.001*dt_fund[0]
	pontos_por_ciclo = np.where((t>=tlim1) & (t<tlim2))[0]
	pontos_por_ciclo = pontos_por_ciclo.astype(int)[0]
	print(pontos_por_ciclo)

	nr_ciclos_amostra = t[len(t)-1] * freq_fund
	nr_ciclos_amostra = int(np.floor(nr_ciclos_amostra))
	n = pontos_por_ciclo*nr_ciclos_amostra
	n = n.astype(int)

	return [freq_fund, n, pontos_por_ciclo,  nr_ciclos_amostra]

def modulo_e_fase(i, n):
	harms = np.fft.fft(i[0:n])
	timestep = t[2]-t[1]
	freq = np.fft.fftfreq(n, d=timestep)

	limite = int(np.floor(n/2) )
	xx = freq[0:limite]

	# módulo da corrente harmônica
	yy = 2/n*np.abs(harms)[0:limite]
	max_yy = np.round(np.max(abs(yy)),3)
	yy = yy / max_yy

	# ângulo da corrente harmônica
	pontos_com_modulo_nao_significativo = np.where(yy<0.01)[0]
	zz = 180/np.pi*np.angle(harms)[0:limite]
	zz[pontos_com_modulo_nao_significativo] = 0

	return[xx, yy, max_yy, zz]

def reconstrucao(tt, yy, zz, freqs):
	iii = np.zeros((len(yy),len(tt)))
	zz = zz * np.pi/180
	ww = 2*np.pi*freqs
	for k in range(len(yy)):
		iii[k,:] = yy[k]*np.cos(ww[k]*tt[:]+zz[k])
	
	return iii



data_frame = pd.read_excel(r'Passagem por zero.xlsx')
t = data_frame['tempo'].to_numpy()
i = data_frame['corrente'].to_numpy()

freq_fund, n, pontos_por_ciclo,  nr_ciclos_amostra = freq_fundamental(t, i)
nharm = int(pontos_por_ciclo/2)
freqs, yy, max_yy, zz = modulo_e_fase(i, n)
delta_freq = freqs[2] - freqs[1]

tt = t[0:pontos_por_ciclo]
ii = reconstrucao(tt, yy, zz, freqs)


fig_3d = plt.figure(figsize=(10,8))
ax_3d = plt.axes(projection='3d')
soma = np.zeros(len(tt))
for k in range(len(yy)):
	if yy[k]>0.01:
		soma = soma + ii[k,:]
		ax_3d.plot3D(1e3*tt[:], (k*delta_freq*tt/tt)/freq_fund, ii[k,:])
		ax_3d.plot3D(1e3*tt[pontos_por_ciclo-1]*freqs/freqs, freqs/freq_fund, yy, color='black')

ax_3d.plot3D(1e3*tt[:], -tt/tt, soma[:], label="total")		
#ax_3d.set_xlim(0, 1e3/freq_fund)
ax_3d.set_xlabel('Tempo [ms]')
ax_3d.set_ylabel('Harmônico [h]')
ax_3d.set_zlabel('Corrente [A]')
ax_3d.set_ylim(-1,15)


plt.show()




