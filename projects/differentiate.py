import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    d = np.zeros_like(u)
    for i in range(len(u)):
        if i == 0:
            d[i] = (u[1] - u[0]) / dt
        elif i == len(u)-1:
         
            d[i] = (u[i] - u[i - 1]) / dt
        else:
        
            d[i] = (u[i + 1] - u[i - 1]) / (2 * dt)
    print(d)
    return d 

def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    d= np.zeros_like(u)
    N_t=len(u)-1
    d[0] = (u[1] - u[0]) / dt
    d[1:N_t]=(u[2:N_t+1] -u[0:N_t-1])/(2*dt)
    d[N_t]=(u[N_t]-u[N_t-1])/dt
    print(d)
    return d

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
    