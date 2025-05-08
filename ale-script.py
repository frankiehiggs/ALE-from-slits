import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve
from scipy.optimize import fsolve
from tqdm import trange
# from numba import jit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from colorspace import hcl_palettes
from datetime import datetime

def wstar(zs, betas):
    """
    Computes the w^* from Pearce, 1991.
    We can use this to find the value of c.
    
    zs    is a list of n points on the circle,
          with the last point equal to 1 + 0j;
    betas is a list of n exponents.

    This involves computing an integral numerically
    with a singularity potentially at one end of the range
    (in the centre of the disc).
    Since we chose a tip as w_n with z_n = 1 and beta_n = -1,
    the boundary has a zero instead of a pole.

    We can handle this using scipy.integrate.quad
    with weight='alg'.
    """
    def unweighted_integrand(t):
        prod = 1
        for i,z in enumerate(zs):
            prod *= (1 - z.conjugate()*t)**(-betas[i])
        return (prod - 1)/t
    integral_re = quad(lambda t : unweighted_integrand(t).real, 0, 1)
    integral_im = quad(lambda t : unweighted_integrand(t).imag, 0, 1)
    # print(f"Computed wstar with error value {integral[1]}")
    return np.exp(integral_re[0] + 1j*integral_im[0])

def ftip(preimage,zs,betas,c):
    """
    Computes the image of `preimage` when `preimage` is
    the preimage of a tip. It's the same method as when
    it's a base, but we do something a slightly different
    to deal with the pole for bases.
    """
    def unweighted_integrand(t):
        prod = 1
        for i,z in enumerate(zs):
            prod *= (1 - z.conjugate()*preimage*t)**(-betas[i])
        return (prod - 1)/t
    integral_re = quad(lambda t : unweighted_integrand(t).real, 0, 1)
    integral_im = quad(lambda t : unweighted_integrand(t).imag, 0, 1)
    return c*preimage*np.exp(integral_re[0] + 1j*integral_im[0])

def fbase(preimage_index,zs,betas,c):
    """
    Computes the image of `preimage` when `preimage` is
    the preimage of a base. Deals carefully with the pole
    at that base itself.
    We take the index instead of the value because then
    we can take the singular term out of the product and
    deal with it separately.
    """
    preimage = zs[preimage_index]
    def unweighted_integrand_1(t):
        # For t in (0,1/2]
        prod = 1
        for i in range(len(zs)):
            prod *= (1 - zs[i].conjugate()*preimage*t)**(-betas[i])
        return (prod - 1)/t
    def unweighted_integrand_2(t):
        # For t in [1/2, 1).
        # The other term becomes (1 - t)^{-1/2}.
        prod = 1
        for i in range(len(zs)):
            if i == preimage_index:
                continue
            else:
                prod *= (1 - zs[i].conjugate()*preimage*t)**(-betas[i])
        return prod/t
    integral1_re = quad(lambda t : unweighted_integrand_1(t).real,0,1/2)
    integral1_im = quad(lambda t : unweighted_integrand_1(t).imag,0,1/2)
    integral1 = integral1_re[0] + 1j*integral1_im[0]
    integral2_re = quad(lambda t : unweighted_integrand_2(t).real,1/2,1,weight='alg',wvar=(0,-1/2))
    integral2_im = quad(lambda t : unweighted_integrand_2(t).imag,1/2,1,weight='alg',wvar=(0,-1/2))
    integral2 = integral2_re[0] + 1j*integral2_im[0]
    integral = integral1 + integral2 - np.log(2)
    return c * preimage * np.exp( integral )

def get_params(ys):
    """
    In the words of Trefethen, this is "easy to do, though not immediate".
    It involves defining a set of linear equations then solving them.
    """
    eys = np.exp(ys)
    # Construct a matrix
    A = np.zeros((len(ys),len(ys)))
    A[0,0] = -(1 + eys[0])
    A[0,1] = eys[0]
    for i in range(1,len(ys)-1):
        A[i,i-1] = 1
        A[i,i]   = -(1 + eys[i])
        A[i,i+1] = eys[i]
    A[len(ys)-1,len(ys)-2] = 1
    A[len(ys)-1,len(ys)-1] = -(1+eys[len(eys)-1])
    b = np.zeros(len(ys))
    b[len(ys)-1] = 2*np.pi*eys[len(eys)-1]
    phis = np.append(solve(A,b),0.0)
    zs = np.exp( 1j * phis )
    return zs

def get_w_hats(zs,betas,c):
    w_hats = np.empty(len(zs),dtype=np.complex128)
    preimage_type = 0 # 0 and 1 for bases, 2 for a tip.
    # The final wn is a tip, so the first two must be bases.
    for i in range(len(zs)):
        if preimage_type == 2:
            # We have a tip
            w_hats[i] = ftip(zs[i],zs,betas,c)
            preimage_type = 0
        else:
            # We have a base
            w_hats[i] = fbase(i,zs,betas,c)
            preimage_type += 1
    return w_hats

def my_arg(z):
    # argument in the range [0,2pi)
    theta = np.angle(z)
    if theta < 0:
        theta += 2*np.pi
    return theta

def objective_fn(ys,betas,ws):
    zs = get_params(ys)
    c = ws[-1]/wstar(zs,betas)
    w_hats = get_w_hats(zs,betas,c)
    equations = np.zeros(len(ys))
    preimage_type = 0 # 0 and 1 for bases, 2 for a tip.
    # The final wn is a tip, so the first two must be bases.
    # If preimage_type is 0 or 2 we have an equation like (3a),
    # and if preimage_type == 1 we have an equation like (3c).
    for i in range(len(ys)-1):
        if preimage_type == 0:
            # w_m is a base, w_{m-1} is a tip
            equations[i] = np.abs(w_hats[i]) - np.abs(ws[i])
            preimage_type += 1
        elif preimage_type == 1:
            # w_m is a base, w_{n-1} is also a base.
            equations[i] = my_arg(w_hats[i]/w_hats[i-1]) - my_arg(ws[i]/ws[i-1])
            preimage_type += 1
        else:
            # w_m is a tip, w_{m-1} is a base
            equations[i] = np.abs(w_hats[i]) - np.abs(ws[i])
            preimage_type = 0
    angle_sum = np.sum( [betas[i]*my_arg(zs[i]) for i in range(len(zs))] )
    remainder = angle_sum % np.pi
    if remainder > 0.5*np.pi:
        remainder = remainder - np.pi
    equations[len(ys)-1] = remainder
    return equations
    
# @jit(nopython=True)
def slitmap(z,theta,capacity):
    ec = np.exp(capacity)
    w = z * np.exp(-1j*theta)
    return np.exp(1j*theta) * ((0.5*ec/w)*(w+1)**2 * (1 + np.sqrt((w*w + 2*(1 - 2/ec)*w + 1)/((w+1)**2))) - 1)

# @jit(nopython=True)
def slitmap_derivative(z,theta,capacity,fz):
    cosbetac = np.cos(np.asin(2*np.exp(-capacity)*np.sqrt(np.exp(capacity)-1)))
    w = z*np.exp(-1j*theta)
    return (fz/z) * (w-1) / np.sqrt(w*w - 2*cosbetac*w + 1)

# @jit(nopython=True)
def slitmap_derivative_reciprocal(z,theta,capacity,fz):
    cosbetac = np.cos(np.asin(2*np.exp(-capacity)*np.sqrt(np.exp(capacity)-1)))
    w = z*np.exp(-1j*theta)
    return (z/fz) * np.sqrt(w*w - 2*cosbetac*w + 1) / (w-1)

def get_random_configuration():
    # T not too large, either 1 or 2.
    T_options = [1,2]
    T_probs   = [0.5,0.5]
    T = np.random.choice(T_options,p=T_probs)
    # n either 10, 100, or 1000, mostly 100
    n_options = [10,100,1000]
    n_probs   = [0.15,0.8,0.05]
    n = np.random.choice(n_options,p=n_probs)
    # eta either 1.5, 2.5 or 10, mostly 2.5
    eta_options = [1.5,2.5,10]
    eta_probs   = [0.1,0.7,0.2]
    eta = np.random.choice(eta_options,p=eta_probs)
    # number of slits between 2 and 6
    slits_options = [2,3,4,5,6]
    slits_probs   = [0.1,0.3,0.1,0.3,0.2]
    slits = np.random.choice(slits_options,p=slits_probs)
    # locations uniformly random
    locations = np.sort(360*np.random.random(slits))
    locations -= min(locations)
    # Each slit has an exponential random length
    lengths = np.ones(slits)
    for i in range(slits):
        l = np.random.choice([0.5,1,2,5,10],p=[0.1,0.5,0.2,0.15,0.05])
        lengths[i] = l
    sigma = (T/n)**8
    return locations,lengths,T,n,eta,sigma

if __name__=='__main__':
    ## Specify these yourself:
    # locations = [0,10] # In degrees, the first has to be 0.
    # lengths   = [2, 1]
    # T = 1.2
    # n = 100
    # eta = 3.00
    # sigma = (T/n)**8
    locations, lengths, T, n, eta, sigma = get_random_configuration()
    print(locations, lengths, T, n, eta, sigma)
    
    circle_pts = 1000
    
    # Derived quantities
    capacity = T/n
    ec = np.exp(capacity)
    d = 2*(ec - 1) + 2*ec*np.sqrt(1 - 1/ec)
    locations = [a*np.pi/180 for a in locations]
    
    ws = np.empty(3*len(locations),dtype=np.complex128)
    ws[0] = np.exp(-1j*locations[0])
    for i in range(1,len(locations)):
        ws[3*i - 2] = np.exp(-1j*locations[i])
        ws[3*i - 1] = np.exp(-1j*locations[i]) / (1 + lengths[i])
        ws[3*i]     = np.exp(-1j*locations[i])
    ws[3*len(locations)-2] = np.exp(-1j*locations[0])
    ws[3*len(locations)-1] = np.exp(-1j*locations[0]) / (1 + lengths[0])
    
    betas = np.empty(3*len(locations))
    preimage_type = 0 # 0 and 1 are bases, 2 is a tip.
    for i in range(3*len(locations)):
        if preimage_type == 2:
            betas[i] = -1
            preimage_type = 0
        else:
            betas[i] = 1/2
            preimage_type += 1
    
    # Get the parameters.
    ys = fsolve(objective_fn,np.zeros(len(ws)-1),args=(betas,ws))
    zs = get_params(ys)
    c = ws[-1] / wstar(zs,betas)
    print("Below are the errors in the tips and bases for our computed map:")
    print(objective_fn(ys,betas,ws))
    
    print("Now we'll simulate the ALE!")
    
    # All of ws, zs, betas and c are global variables.
    def phi0(z):
        # Takes z in the exterior disc
        # to its image under phi0.
        return 1/ftip(1/z,zs,betas,c)
    
    #@jit(nopython=True)
    def phi0_prime(z,image):
        # Does it make more sense when eta > 0 to compute the
        # reciprocal of the derivative instead?
        derivative = image/z
        for m in range(len(zs)):
            derivative *= (z - zs[m].conjugate())**(-betas[m])
        return derivative
    
    #@jit(nopython=True)
    def phi0_prime_reciprocal(z,image):
        output = z/image
        for m in range(len(zs)):
            output *= (z - zs[m].conjugate())**(betas[m])
        return output

    angle_seq = np.empty(shape=n)
    raw_candidates = np.linspace(0,2*np.pi,num=circle_pts,endpoint=False)
    for t in trange(n):
        # We rotate the candidate points through a random angle
        # to avoid evaluating the derivative exactly at the tips,
        # since when eta > 0 that would mean dividing by 0.
        candidates = (raw_candidates + 2*np.pi*np.random.random()) % (2*np.pi)
        z_candidates = np.exp( sigma + 1j*candidates )
        images = np.empty(shape=len(candidates),dtype=np.complex128)
        derivs = np.ones(shape=len(candidates))
        for i in range(len(candidates)):
            images[i] = z_candidates[i]
            for s in reversed(range(t)):
                new_loc = slitmap(images[i],angle_seq[s],capacity)
                derivs[i] *= np.abs(slitmap_derivative_reciprocal(images[i],angle_seq[s],capacity,new_loc))
                images[i] = new_loc
            derivs[i] *= np.abs(phi0_prime_reciprocal(images[i],phi0(images[i])))
        derivs = np.nan_to_num(derivs**eta)
        # print(sum(derivs))
        chosen_index = np.random.choice(circle_pts,p=derivs/sum(derivs))
        angle_seq[t] = candidates[chosen_index]
    
    my_palette = hcl_palettes().get_palette(name="Batlow")
    colours = my_palette.colors(int(n))
    
    slitdensity = 100 # points plotted per slit.
    slit = np.linspace(1+d,1,num=slitdensity,endpoint=False)
    
    cluster = np.empty(shape=slitdensity*n,dtype=np.complex128)
    for t in range(n):
        cluster[:t*slitdensity] = slitmap(cluster[:t*slitdensity],angle_seq[n-1-t],capacity)
        cluster[t*slitdensity:(t+1)*slitdensity] = np.exp(1j*angle_seq[n-1-t])*slit
    for i in trange(len(cluster)):
        cluster[i] = phi0(cluster[i])
    
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    unit_circle = Circle((0,0),1,fill=False)
    ax.add_patch(unit_circle)
    for k in range(len(locations)):
        ax.plot([np.cos(locations[k]), (1+lengths[k])*np.cos(locations[k])],[np.sin(locations[k]), (1+lengths[k])*np.sin(locations[k])],
                linewidth=1, color='k', marker="none")
    for t in range(n):
        ax.scatter(cluster.real[t*slitdensity:(t+1)*slitdensity],cluster.imag[t*slitdensity:(t+1)*slitdensity],
                   color=colours[t],s=1)
    
    now = datetime.now().isoformat()
    fig.savefig(f"{now}.pdf")
    with open(f"{now}.txt","w") as configfile:
        locations, lengths, T, n, eta, sigma
        configfile.write("locations = ")
        configfile.write(str([float(theta*180/np.pi) for theta in locations]))
        configfile.write("\nlengths = ")
        configfile.write(str([float(l) for l in lengths]))
        configfile.write("\nT = ")
        configfile.write(str(T))
        configfile.write("\nn = ")
        configfile.write(str(n))
        configfile.write("\neta = ")
        configfile.write(str(eta))
        configfile.write("\nsigma = ")
        configfile.write(str(sigma))
        configfile.write("\n")
