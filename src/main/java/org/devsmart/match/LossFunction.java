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
            return Math.sqrt( error.getSumsq() / error.getN() );
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
                final double x1 = x[i] == 0.0 ? 1e-4 : x[i];
                final double y1 = y[i] == 0.0 ? 1e-4 : y[i];
                sum += x1*Math.log(y1) + (1-x1)*Math.log(1-y1);
            }
            return -sum;
        }
    };
}
