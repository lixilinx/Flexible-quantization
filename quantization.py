import numpy as np

def quantizer(x, nbits=8, blk_exp=False, sym=False, stochastic=False):
    """
    Quantize numpy array x to n-bits int as
        x => round(a*x + b)
    or approximates x as 
        x ~ (round(a*x + b) - b)/a
        
    If blk_exp = True, log2(a) will be an int, otherwise, a float.
    
    If sym = True, b = 0, otherwise, an int.  
    
    If stochastic = True (use only in quantization aware training), expectation of quantized x equals x, otherwise, biased (used for inference).  
    """
    if sym: # symmetric
        x1 = np.max(np.abs(x))
        x0 = -x1
    else: # asymmetric
        x0, x1 = np.min(x), np.max(x)
        
    a = (2**nbits - 1)/(x1 - x0)
    if blk_exp: # block exponent
        a = 2**np.floor(np.log2(a))       
        
    b = np.floor(-a*(x0 + x1)/2) # zero point always is an int
    
    if stochastic:
        qx = np.round(a*x + b + np.random.uniform(size=x.shape) - 0.5)
    else:
        qx = np.round(a*x + b + 1) - 1 
        # why not just round(a*x + b):
        #   numpy rounds (int+0.5) to the nearest even number (compatible with IEEE standards).
        #   Hence, say for int8 and a*x+b=127.5, round(127.5)=128 (overflow), while round(127.5+1)-1 = 127 (desired). 

    return qx, a, b # quantized x is (qx - b)/a


if __name__ == '__main__':
    # some examples
    from matplotlib import pyplot as plt 
    
    x = np.random.normal(size=(2, 3, 5))
    
    qx, a, b = quantizer(x, blk_exp=False, sym=False)
    assert np.max(qx)<=127 and np.min(qx)>=-128
    print('max abs quantization err: {}'.format(np.max(np.abs(x - (qx-b)/a))))
    
    qx, a, b = quantizer(x, blk_exp=False, sym=True)
    assert np.max(qx)<=127 and np.min(qx)>=-128
    print('max abs quantization err: {}'.format(np.max(np.abs(x - (qx-b)/a))))
    
    qx, a, b = quantizer(x, blk_exp=True, sym=False)
    assert np.max(qx)<=127 and np.min(qx)>=-128
    print('max abs quantization err: {}'.format(np.max(np.abs(x - (qx-b)/a))))

    qx, a, b = quantizer(x, blk_exp=True, sym=True)
    assert np.max(qx)<=127 and np.min(qx)>=-128
    print('max abs quantization err: {}'.format(np.max(np.abs(x - (qx-b)/a))))
    
    # the stochastic version always is unbiased
    for blk_exp in [False, True]:
        for sym in [False, True]:
            expectation_qx = np.zeros(x.shape)
            stds = []
            for trial in range(1000):
                qx, a, b = quantizer(x, nbits=8, blk_exp=blk_exp, sym=sym, stochastic=True)
                expectation_qx = (trial*expectation_qx + qx)/(trial + 1)
                stds.append(np.std(x - (expectation_qx-b)/a))
            plt.loglog(stds)
    plt.ylabel('STD(real - E[quantized])')
    plt.xlabel('Monte Carlo trials')
    plt.title('Stochastic quantization of Normal(0, 1) numbers')
    plt.legend(['Float-Asym', 'Float-Sym', 'BlkExp-Asym', 'BlkExp-Sym'])