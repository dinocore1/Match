package org.devsmart.match.rbm.nuron;


import org.apache.commons.math3.util.FastMath;

public class SoftPlus implements Neuron {


    @Override
    public double value(double input) {
        return FastMath.log(1 + FastMath.exp(input));
    }

    @Override
    public double derivative(double input) {
        return 1 / (1 + FastMath.exp(-input));
    }
}
