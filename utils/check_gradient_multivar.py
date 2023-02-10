import numpy as np
import matplotlib.pyplot as plt

''' Function to be evaluated '''
def f(q):
    return np.array([q[0]**3,3*q[1]**2])

''' First derivative of f(q) '''
def df(q):
    return np.array([3*q[0]**2,6*q[1]])

''' How to compute x axis values '''
def xAxis(q,dq,ε):
    return np.array([np.log(ε[i][0]) for i in range(10)])

''' How to compute y axis values '''
def yAxis(q,dq,ε):
    return np.array([np.log(np.linalg.norm(f(q+ε[i]*dq) - f(q) - df(q)*ε[i]*dq)) for i in range(10)])

''' Calculate Slope '''
def slope(x,y):
    return np.polyfit(x, y, 1)[0]

''' Check that gradient slope is correct '''
def main():
    # Get dq something small
    dq = np.array([1e-6, 1e-6])

    # Get a range of ε values to evaluate the gradient at
    ε  = [np.array([e+1,e+1]) for e in range(10)]

    # Evaluate at different points
    for c in np.linspace(1,10,10):
        q = np.array([c,c])
        
        # Get the x,y axis values for the plot
        x  = xAxis(q,dq,ε)
        y  = yAxis(q,dq,ε)

        # Check the slope
        s = slope(x,y)

        # Plot
        plt.plot(xAxis(q,dq,ε), yAxis(q,dq,ε), label=f'at q={q}, slope={s:0.2f}')
    
    # plt.rcParams.update({
    #     "text.usetex": True
    # })

    plt.title('$f(q)=||f(q+\\epsilon{}dq)-f(q)-\\nabla{}f|_{q}\\epsilon{}dq||$')
    plt.legend(loc="upper left")
    plt.xlabel('$log(\\epsilon{})$')
    plt.ylabel('$log(f(q))$')
    plt.show()

if __name__ == "__main__":
    main()