package org.devsmart.match.rbm;


import java.util.Random;

public class GibbsSample {

    public static double[] sample(RBM rbm, double[] input, int numSteps, Random r) {
        for(int i=0;i<numSteps;i++){
            double[] hidden = rbm.activateHidden(input, r);
            input = rbm.activateVisible(hidden, r);
        }
        return input;
    }
}
