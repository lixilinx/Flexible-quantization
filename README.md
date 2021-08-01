# A simple formula for eight types of quantization
We quantize a block of real numbers *x* into integers of *n*-bits per number as 
 
 float --> int: *x* --> round(*ax* + *b*)
 
 This sample code has implemented three sets of independent quantization controls:
 
 1, *a* is an ordinary floating point number, or special such that log2(*a*) is an integer (block exponent). Using block exponent can completely eliminate the need of floating point numbers. 
 
 2, *b* is an ordinary integer, or set to zero (symmetric range). In many situations, the dynamic ranges of *x* are almost symmetric. Thus, fixing *b*=0 simplifies the calculations a lot, while without causing much performance loss. 
 
 3, stochastic or deterministic quantization. The stochastic quantization is unbiased, and it could improve the performance of straight through quantization aware training (QAT). Actually, for a quadratic cost, the gradients of stochastically quantized coefficients are unbiased as well, while the ones of deterministically quantized coefficients are biased.  
 
 Totally, we get eight types of quantizations as defined in the Numpy sample code. Just one more note: in the actually implementation, we do need to round *ax*+*b* as round(*ax* + *b* + 1) - 1 (see the comments for details). 
 
 ![Alt Image text](https://github.com/lixilinx/Flexible-quantization/blob/main/sample.png)
 
 Fig. 1, a demo shows that the stochastic quantization is asymptotically unbiased for all types of *a* and *b*. 
 
 ### Refs:
 
Details of QAT in Tensorflow and Pytorch can be found at [here](https://arxiv.org/pdf/1712.05877.pdf) and [here](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/). Seems both only consider the case of (non-block exponent, asymmetric range, deterministic quantization), and not flexible enough for me. Stochastic quantization, e.g., discussed [here](https://arxiv.org/pdf/2006.10159v1.pdf), might decrease the gradient bias of straight through QAT. At least for a quadratic cost, the gradients of stochastically quantized coefficients are unbiased.
