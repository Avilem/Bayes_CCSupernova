
import numpy as np

# -------------------------------

# Create a Core Bounce Signal Model with four parameters $h(t;\beta,\alpha,\tau,s)$

def CoreBounceSignal(t,beta, alpha, tau,s,D):
    """
    t: numpy array for time.
    beta: rotational rate of the core
    alpha: EOS parameter (height of the third peak)
    tau: time at which the core-bounce occurs
    D: Distance to the source in meters

    --Parameters suggested ranges --
    alpha \in [1, 380]

    beta  \in [0.005, 0.12]

    tau   \in [-6, -1] * 1e-4

    s \in [1.8, 2.32] e-4

    """


    # Peak's amplitudes

    coef1 = np.array( [-1.3226599612574933e+01,  2.895774762651694e+03, -1.3187237843459969e+04,  0.00000000e+00])

    coef2 = np.array([ -1.037868009414148e+00, -5.524811161263574e+03,  9.436966253755723e+03,  0.00000000e+00])

    p1    = coef1[0] + coef1[1] * beta + coef1[2] * beta**2

    p2    = coef2[0] + coef2[1] * beta + coef2[2] * beta**2

    p3    = (17.20  + alpha*(beta/0.06)**2)



    # -------------------------------

    # Time shift (mu) of each term

    mu1    = tau

    mu2    = mu1 + 0.0005

    mu3    = mu2 + 0.0005



    # -------------------------------

    # Compute gaussian terms


    g1    = p1 * np.exp( -(t-mu1)**2/(2*s**2) )

    g2    = p2 * np.exp( -(t-mu2)**2/(2*s**2) )

    g3    = p3 * np.exp( -(t-mu3)**2/(2*s**2) )



    # -------------------------------

    # Compute signal

    h     = g1 + g2 + g3
    #Return h/(100*D) to convert from centimeters to meters with distance D
    return h/(100*D)


