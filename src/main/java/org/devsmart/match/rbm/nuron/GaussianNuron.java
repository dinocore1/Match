package org.devsmart.match.rbm.nuron;

import org.apache.commons.math3.analysis.function.Gaussian;

public class GaussianNuron implements Nuron {

    private double mean = 0;
    private double sigma = 1;
    private Gaussian gaussian = new Gaussian(0, 1);

    public GaussianNuron() {

    }

    public void setParams(double mean, double sigma) {
        this.mean = mean;
        this.sigma = sigma;
        this.gaussian = new Gaussian(mean, sigma);
    }

    public double getMean() {
        return mean;
    }

    public double getSigma() {
        return sigma;
    }

    @Override
    public double activate(double input) {
        double retval = gaussian.value(input);
        return retval;
    }
}
