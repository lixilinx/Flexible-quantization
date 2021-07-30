# Flexible-quantization
A simple formula supports four types of quantization: float or block exponent scaling, symmetric or asymmetric range

We may need to quantization the coefficients or activiations of a Pytorch or Tensorflow models for quantization aware training or deployment on resource limited devices. Let the dynamic range of a float number x be [x0, x1], i.e., x0 <= x <= x1. We want to represent x as an n-bits of integers as

q = round(a*x + b)

where a can be an ordinary floating point number or special such as log2(a) is an int (block exponent), b can be an ordinary int or zero. Hence, totally there are types of quantization schemes. Note that b always should be an int such that no quantization error arises for x=0. 


