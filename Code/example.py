import numpy as np

def compute_pi(n):
    """
    Compute pi using Leibniz' formula:
    1 - 1/3 + 1/5 - 1/7 + 1/9 - ... = pi/4
    """
    k = 1   # Denominator
    s = 0   # Sum
     
    for i in range(n):
        if i % 2 == 0:
            s += 4/k
        else:
            # odd index elements are negative
            s -= 4/k
        k += 2
         
    return s

def main():
    n = 1000
    pi = compute_pi(n)
    pi_ref = np.pi
    print("Pi (computed):   %f (n=%d)" % (pi,n))
    print("Pi (reference):  %f" % pi_ref)
    print("Difference:      %f" % (pi_ref-pi))


main()
