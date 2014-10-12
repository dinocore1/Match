package org.devsmart.match.rbm.nuron;


public class Bias implements Neuron {
    @Override
    public double value(double input) {
        return 1.0;
    }

    @Override
    public double derivative(double input) {
        return 1.0;
    }
}
