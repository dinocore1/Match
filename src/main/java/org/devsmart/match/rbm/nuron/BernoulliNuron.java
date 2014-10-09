package org.devsmart.match.rbm.nuron;


import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.exception.DimensionMismatchException;

public class BernoulliNuron implements Neuron {

    private static Sigmoid sigmoid = new Sigmoid();

    public BernoulliNuron() {

    }

    @Override
    public DerivativeStructure value(DerivativeStructure t) throws DimensionMismatchException {
        return sigmoid.value(t);
    }

    @Override
    public double value(double x) {
        return sigmoid.value(x);
    }
}
