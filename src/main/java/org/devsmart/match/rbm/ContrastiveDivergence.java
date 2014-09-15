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

        double[] hiddenPositive = rbm.sampleHidden(trainingVisible, r);

        double[] input = trainingVisible;
        for(int i=0;i<numGibbsSteps;i++){
            input = rbm.sampleHidden(input, r);
            input = rbm.activateVisible(input);
        }

        double[] negitive = input;
        double[] hiddenNegitive = rbm.sampleHidden(negitive, r);

        final ArrayRealVector trainingVisibleV = new ArrayRealVector(trainingVisible);
        final ArrayRealVector hiddenPositiveV = new ArrayRealVector(hiddenPositive);
        final ArrayRealVector negitiveV = new ArrayRealVector(negitive);
        final ArrayRealVector hiddenNegitiveV = new ArrayRealVector(hiddenNegitive);
        WGradient = trainingVisibleV.outerProduct(hiddenPositiveV).subtract(negitiveV.outerProduct(hiddenNegitiveV));
        AGradient = trainingVisibleV.subtract(negitiveV);
        BGradient = hiddenPositiveV.subtract(hiddenNegitiveV);

    }


}
