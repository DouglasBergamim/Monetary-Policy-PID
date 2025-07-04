class DiscretePIDController:
    """
    A discrete-time PID controller using the Tustin (bilinear) approximation.
    Parameters
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    dt : float
        Sampling period (in seconds).
    """
    def __init__(self, kp:float, ki:float, kd:float, dt:float, u_min:float | None = None, u_max:float | None = None):
        """
        Instantiate a new controller and pre-compute the discrete gains.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.b0 = self.kp + self.ki*self.dt/2 + 2*self.kd/self.dt
        self.b1 = -self.kp + self.ki*self.dt/2 - 4*self.kd/self.dt
        self.b2 = 2*self.kd/self.dt
        self.e_prev = 0.0
        self.e_prev2 = 0.0
        self.u_prev = 0.0   
        self.u_min = u_min
        self.u_max = u_max
        
    def reset(self):
        """
        Reset the integral and derivative history.
        """
        self.e_prev = 0.0
        self.e_prev2 = 0.0
        self.u_prev = 0.0

    def set_parameters(self, kp:float, ki:float, kd:float):
        """
        Change the PID gains at run-time.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.b0 = self.kp + self.ki*self.dt/2 + 2*self.kd/self.dt
        self.b1 = -self.kp + self.ki*self.dt/2 - 4*self.kd/self.dt
        self.b2 = 2*self.kd/self.dt
        self.reset()
    
    def step(self, error:float):
        """
        Compute the control signal from the current error.
        """        
        u = self.b0*error + self.b1*self.e_prev + self.b2*self.e_prev2 + self.u_prev
        u = self._saturate(u)
        self.e_prev2 = self.e_prev
        self.e_prev = error
        self.u_prev = u
        return u
    
    def _saturate(self, u:float):
        """
        Saturate the control signal to the minimum and maximum values.
        """
        if self.u_min is not None and self.u_max is not None:
            return max(self.u_min, min(u, self.u_max))
        elif self.u_min is not None:
            return max(self.u_min, u)
        elif self.u_max is not None:
            return min(self.u_max, u)
        else:
            return u

# just for testing purposes
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    # parameters
    dt   = 0.01
    pid  = DiscretePIDController(kp=2.0, ki=1.0, kd=0.05,
                                dt=dt, u_min=0.0, u_max=5.0)

    a    = 1.0        
    Tf   = 10.0        
    N    = int(Tf/dt)

    y = 0.0            
    ys, us, rs = [], [], []

    for k in range(N):
        r = 1.0                  
        e = r - y
        u = pid.step(e)
        y = y + dt*(-a*y + a*u) 

        ys.append(y); us.append(u); rs.append(r)

    t = np.arange(N)*dt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(t, ys, label='y')
    plt.plot(t, rs, '--', label='ref')
    plt.title('Output'); plt.grid(); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t, us, 'tab:orange')
    plt.title('Control signal'); plt.grid()
    plt.tight_layout(); plt.show()