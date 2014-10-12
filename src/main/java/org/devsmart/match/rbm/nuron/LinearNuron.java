package org.devsmart.match.rbm.nuron;


public class LinearNuron implements Neuron {
    @Override
    public double value(double input) {
        return input;
    }

    @Override
    public double derivative(double input) {
        return 1.0;
    }
}
