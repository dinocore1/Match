package org.devsmart.match.rbm.nuron;


import org.apache.commons.math3.util.FastMath;

public class SoftPlus implements Neuron {


    @Override
    public double value(double input) {
        final double exp = FastMath.exp(input);
        if(Double.isInfinite(exp)){
            return input;
        } else {
            return FastMath.log(1 + exp);
        }
    }

    @Override
    public double derivative(double input) {
        return 1 / (1 + FastMath.exp(-input));
    }
}
