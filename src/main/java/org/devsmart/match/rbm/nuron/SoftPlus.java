package org.devsmart.match.rbm.nuron;


import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.function.Logistic;
import org.apache.commons.math3.exception.DimensionMismatchException;

public class SoftPlus implements Neuron {

    Logistic logistic = new Logistic();

    public SoftPlus() {

    }

    @Override
    public DerivativeStructure value(DerivativeStructure t) throws DimensionMismatchException {
        t.
        return logistic.value(t);
    }

    @Override
    public double value(double x) {
        return Math.log(1+Math.exp(x));
    }
}
