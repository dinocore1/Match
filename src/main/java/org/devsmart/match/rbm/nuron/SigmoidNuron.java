package org.devsmart.match.rbm.nuron;


import org.apache.commons.math3.util.FastMath;

public class SigmoidNuron implements Neuron {

    @Override
    public double value(double x) {
        return 1 / (1 + FastMath.exp(-x));
    }

    @Override
    public double derivative(double input) {
        double value = value(input);
        return (1-value)*value;
    }
}
