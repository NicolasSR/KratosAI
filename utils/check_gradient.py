import math
import numpy as np
import matplotlib.pyplot as plt

class GradientCheck():
    def __init__(self,f,df):
        self.f = f
        self.df = df

    def eval_df(self, q):    
        return self.df.T

    def xAxis(self,q,dq,ε):
        return np.array([np.log(ε[i]) for i in range(10)])

    def yAxis(self,q,dq,ε):
        
        res = []
        for e in ε:
            dq_effective = e*dq
            q_perturbed = q + dq_effective
            f_q = self.f(q)
            f_q_perturbed = self.f(q_perturbed)
            r = np.linalg.norm( f_q_perturbed- f_q - dq_effective @ self.df.T)
            
            print("for eps = ,",e," r=",r, " f(q)=",np.linalg.norm(f_q-f_q_perturbed) )
            res.append(math.log(r))
        # return np.array([np.log(np.linalg.norm(self.f(q+ε[i]*dq)-self.f(q)-self.eval_df(q))) for i in range(10)])
        return res

    ''' Calculate Slope '''
    def slope(self,x,y):
        print(f"{x=}")
        print(f"{y=}")
        return np.polyfit(x, y, 1)[0]

    def do_check(self,q):
        # Get dq something small
        dq  = np.random.normal(0, 1, size=q.shape)
        dq /= np.linalg.norm(dq)

        # Get a range of ε values to evaluate the gradient at
        # ε =  [np.full(q.shape, e+1) for e in range(10)]
        # εr = [np.full((1,3), e+1) for e in range(10)]
        ε = [1e-1/(2**(1+e)) for e in range(10)]

        # Get the x,y axis values for the plot
        x  = self.xAxis(q,dq,ε)
        y  = self.yAxis(q,dq,ε)

        # Check the slope
        s = self.slope(x,y)

        # Plot
        plt.plot(x, y, label=f'at q, slope={s:0.2f}')

        plt.title('$f(q)=||f(q+\\epsilon{}dq)-f(q)-\\nabla{}f|_{q}\\epsilon{}dq||$')
        plt.legend(loc="upper left")
        plt.xlabel('$log(\\epsilon{})$')
        plt.ylabel('$log(f(q))$')
        plt.show()