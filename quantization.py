import numpy as np

def quantizer(x, nbits=8, blk_exp=False, sym=False):
    """
    Quantize numpy array x to representation with n-bits of precisions as
        x => round(a*x + b)
    or 
        x ~ (round(a*x + b) - b)/a
        
    If blk_exp = True, log2(a) will be an int, otherwise, a float.
    If sym = True, b = 0, otherwise, an int.     
    """
    if sym: # symmetric
        x1 = np.max(np.abs(x))
        x0 = -x1
    else: # asymmetric
        x0 = np.min(x)
        x1 = np.max(x)
        
    if blk_exp: # block exponent
        a = 2**np.floor(np.log2((2**nbits - 1)/(x1 - x0)))
    else: # float scale
        a = (2**nbits - 1)/(x1 - x0)
        
    b = np.floor(-a*(x0 + x1)/2) # zero point always is an int
    
    qx = np.round(a*x + b + 1) - 1 # int representation 
        # why +1 then -1? np rounds to the nearest even number (compatible with IEEE).
        # Hence, say for int8, this will round 127.5 to 127, not 128 (it overflows). 

    return qx, a, b # quantized representation as x ~ (qx - b)/a


if __name__ == '__main__':
    x = np.random.normal(size=(2, 3, 5)) + np.random.randn()
    
    qx, a, b = quantizer(x, blk_exp=False, sym=False)
    assert np.max(qx)<=127 and np.min(qx)>=-128
    print('max quantization err: {}'.format(np.max(np.abs(x - (qx-b)/a))))
    
    qx, a, b = quantizer(x, blk_exp=False, sym=True)
    assert np.max(qx)<=127 and np.min(qx)>=-128
    print('max quantization err: {}'.format(np.max(np.abs(x - (qx-b)/a))))
    
    qx, a, b = quantizer(x, blk_exp=True, sym=False)
    assert np.max(qx)<=127 and np.min(qx)>=-128
    print('max quantization err: {}'.format(np.max(np.abs(x - (qx-b)/a))))

    qx, a, b = quantizer(x, blk_exp=True, sym=True)
    assert np.max(qx)<=127 and np.min(qx)>=-128
    print('max quantization err: {}'.format(np.max(np.abs(x - (qx-b)/a))))