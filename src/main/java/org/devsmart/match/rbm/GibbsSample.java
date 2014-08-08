package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.RealVector;

import java.util.Random;

public class GibbsSample {

    public static RealVector sample(RBM rbm, RealVector input, int numSteps, Random r) {
        for(int i=0;i<numSteps;i++){
            RealVector hidden = rbm.activateHidden(input, r);
            input = rbm.activateVisible(hidden, r);
        }
        return input;
    }
}
