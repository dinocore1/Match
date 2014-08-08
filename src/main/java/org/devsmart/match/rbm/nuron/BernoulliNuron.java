package org.devsmart.match.rbm.nuron;


import org.apache.commons.math3.analysis.function.Sigmoid;

public class BernoulliNuron implements Nuron {

    private static Sigmoid sigmoid = new Sigmoid();

    public BernoulliNuron() {

    }

    @Override
    public double activate(double input) {
        double retval = sigmoid.value(input);
        return retval;
    }
}
