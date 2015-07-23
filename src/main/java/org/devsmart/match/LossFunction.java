package org.devsmart.match;


public interface LossFunction {

    double loss(double[] x, double[] y);

    LossFunction Square = new LossFunction() {

        @Override
        public double loss(double[] x, double[] y) {
            if(x.length != y.length) {
                throw new IllegalArgumentException("array sizes must be equal");
            }
            double retval = 0;
            for(int i=0;i<x.length;i++){
                retval += (x[i] - y[i]) * (x[i] - y[i]);
            }
            return -retval;
        }
    };

    LossFunction CrossEntropy = new LossFunction() {
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
