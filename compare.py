import numpy as np
import termplotlib as tpl

steps  = 100

fom    = np.load("FOM.npy")
nn_rom = np.load("NNM.npy")

nn_rom = nn_rom[:,:] # Shift all results because the fom has the results
fom    = fom[:,:]     # at the end of the step and pinn at the begining

print("Error plot:")
x = np.linspace(0, steps, steps)
y = np.array([np.linalg.norm(fom[i]-nn_rom[i])/np.linalg.norm(fom[i]) for i in range(steps)])
fig = tpl.figure()
fig.plot(x, y, width=75, height=35)
fig.show()

print("======================")
t = [np.max(fom[i]-nn_rom[i]) for i in range(steps)]
for i in range(steps):
    # This is for debug
    if False and (i == 0 or i == steps-1):
        print(f"{i=:2d}: {np.linalg.norm(fom[i]-nn_rom[i])=}, {np.linalg.norm(fom[i])=}")
        print("======================")
    if not i % 10:
        print(f"Error T={i:2d}: {t[i]:.8f}")

print("======================")
print(f"Error ALL : {np.linalg.norm(fom-nn_rom)/np.linalg.norm(fom):.8f}")
