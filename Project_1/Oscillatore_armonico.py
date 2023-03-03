import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

class App:
    def __init__(self, master):
        self.master = master
        master.title("Oscillatore armonico forzato e smorzato")

        # Parametri dell'oscillatore
        self.k = 1
        self.m = 1
        self.gamma = 0.1
        self.F = 1
        self.omega = 1

        # Interfaccia grafica
        tk.Label(master, text="k:").grid(row=0, column=0)
        tk.Label(master, text="m:").grid(row=1, column=0)
        tk.Label(master, text="gamma:").grid(row=2, column=0)
        tk.Label(master, text="F:").grid(row=3, column=0)
        tk.Label(master, text="omega:").grid(row=4, column=0)

        self.k_entry = tk.Entry(master)
        self.k_entry.insert(0, str(self.k))
        self.k_entry.grid(row=0, column=1)

        self.m_entry = tk.Entry(master)
        self.m_entry.insert(0, str(self.m))
        self.m_entry.grid(row=1, column=1)

        self.gamma_entry = tk.Entry(master)
        self.gamma_entry.insert(0, str(self.gamma))
        self.gamma_entry.grid(row=2, column=1)

        self.F_entry = tk.Entry(master)
        self.F_entry.insert(0, str(self.F))
        self.F_entry.grid(row=3, column=1)

        self.omega_entry = tk.Entry(master)
        self.omega_entry.insert(0, str(self.omega))
        self.omega_entry.grid(row=4, column=1)

        self.plot_button = tk.Button(master, text="Plot", command=self.plot)
        self.plot_button.grid(row=5, column=0, columnspan=2)

    def plot(self):
        # Leggi i parametri dall'interfaccia grafica
        self.k = float(self.k_entry.get())
        self.m = float(self.m_entry.get())
        self.gamma = float(self.gamma_entry.get())
        self.F = float(self.F_entry.get())
        self.omega = float(self.omega_entry.get())

        # Calcola l'oscillazione forzata e smorzata
        t = np.linspace(0, 10*np.pi, 1000)
        x = np.exp(-self.gamma/2 * t) * (self.F/self.m - self.k/self.m * (1j*self.omega*t - self.gamma/2) / ((1j*self.omega)**2 - self.gamma**2/4))

        # Disegna il grafico
        fig = plt.figure()
        plt.plot(t, np.real(x), label="Posizione")
        plt.plot(t, np.imag(x), label="Velocità")
        plt.xlabel("Tempo")
        plt.legend()
        plt.show()

root = tk.Tk()
app = App(root)
root.mainloop()
