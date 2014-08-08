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

        RealVector input = hiddenPositive;
        for(int i=1;i<numGibbsSteps;i++){
            input = rbm.activateVisible(input, r);
            input = rbm.activateHidden(input, r);
        }
        RealVector negitive = rbm.getVisibleInput(input);

        RealVector hiddenNegitive = rbm.activateHidden(rbm.activateVisible(input, r), r);

        WGradient = trainingVisible.outerProduct(hiddenPositive).subtract(negitive.outerProduct(hiddenNegitive));
        AGradient = trainingVisible.subtract(negitive);
        BGradient = hiddenPositive.subtract(hiddenNegitive);

    }


}
