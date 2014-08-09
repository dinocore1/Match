package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Random;

public class ContrastiveDivergence {

    public RealMatrix WGradient;
    public RealVector AGradient;
    public RealVector BGradient;

    public void train(RBM rbm, double[] trainingVisible, int numGibbsSteps, Random r) {

        double[] hiddenPositive = rbm.activateHidden(trainingVisible, r);

        double[] input = hiddenPositive;
        for(int i=1;i<numGibbsSteps;i++){
            input = rbm.activateVisible(input, r);
            input = rbm.activateHidden(input, r);
        }
        double[] negitive = rbm.getVisibleInput(input);

        double[] hiddenNegitive = rbm.activateHidden(rbm.activateVisible(input, r), r);

        final ArrayRealVector trainingVisibleV = new ArrayRealVector(trainingVisible);
        final ArrayRealVector hiddenPositiveV = new ArrayRealVector(hiddenPositive);
        final ArrayRealVector negitiveV = new ArrayRealVector(negitive);
        final ArrayRealVector hiddenNegitiveV = new ArrayRealVector(hiddenNegitive);
        WGradient = trainingVisibleV.outerProduct(hiddenPositiveV).subtract(negitiveV.outerProduct(hiddenNegitiveV));
        AGradient = trainingVisibleV.subtract(negitiveV);
        BGradient = hiddenPositiveV.subtract(hiddenNegitiveV);

    }


}
