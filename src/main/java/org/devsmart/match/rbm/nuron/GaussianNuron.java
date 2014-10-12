package org.devsmart.match.rbm.nuron;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.function.Gaussian;

public class GaussianNuron implements Neuron {

    private Gaussian gaussian = new Gaussian();

    @Override
    public double value(double input) {
        return gaussian.value(input);
    }

    @Override
    public double derivative(double input) {
        DerivativeStructure x = new DerivativeStructure(1, 1, input);
        return gaussian.value(x).getPartialDerivative(1);
    }
}
