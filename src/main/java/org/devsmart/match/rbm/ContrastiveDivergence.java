package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Random;

public class ContrastiveDivergence {

    public RealMatrix WGradient;
    public RealVector AGradient;
    public RealVector BGradient;

    public void train(RBM rbm, RealVector trainingVisible, int numGibbsSteps, Random r) {

        RealVector hiddenPositive = rbm.activateHidden(trainingVisible, r);
        RealVector input = rbm.activateVisible(hiddenPositive, r);
        for(int i=1;i<numGibbsSteps;i++){
            input = rbm.activateHidden(input, r);
            input = rbm.activateVisible(input, r);
        }
        RealVector negitive = input;
        RealVector hiddenNegitive = rbm.activateHidden(negitive, r);

        WGradient = trainingVisible.outerProduct(hiddenPositive).subtract(negitive.outerProduct(hiddenNegitive));
        AGradient = trainingVisible.subtract(negitive);
        BGradient = hiddenPositive.subtract(hiddenNegitive);

    }


}
