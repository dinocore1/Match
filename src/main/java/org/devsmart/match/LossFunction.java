package org.devsmart.match;


import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

public interface LossFunction {

    double loss(double[] x, double[] y);

    public static final LossFunction SumOfSquares = new LossFunction() {

        @Override
        public double loss(double[] x, double[] y) {
            if(x.length != y.length) {
                throw new IllegalArgumentException("array sizes must be equal");
            }
            SummaryStatistics error = new SummaryStatistics();
            for(int i=0;i<x.length;i++){
                error.addValue(x[i] - y[i]);
            }
            return error.getSumsq();
        }
    };

    public static final LossFunction CrossEntropy = new LossFunction() {
        @Override
        public double loss(double[] x, double[] y) {
            if(x.length != y.length) {
                throw new IllegalArgumentException("array sizes must be equal");
            }
            double sum = 0;
            for(int i=0;i<x.length;i++){
                sum += x[i]*Math.log(y[i]) + (1-x[i])*Math.log(1-y[i]);
            }
            return -sum;
        }
    };
}
