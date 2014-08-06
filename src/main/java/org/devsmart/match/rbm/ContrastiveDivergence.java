package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class ContrastiveDivergence {

    public static RealVector gibbsSample(RBM rbm, RealVector input, int numSteps) {
        for(int k=0;k<numSteps;k++) {
            RealVector hidden = rbm.activateHidden(input);
            input = rbm.activateVisible(hidden);
        }
        return input;
    }

    public RealMatrix WGradient;
    public RealVector AGradient;
    public RealVector BGradient;

    public void train(RBM rbm, RealVector trainingVisible, int numGibbsSteps) {
        RealVector negitive = gibbsSample(rbm, trainingVisible, numGibbsSteps);

        RealVector hiddenPositive = rbm.activateHidden(trainingVisible);
        RealVector hiddenNegitive = rbm.activateHidden(negitive);

        WGradient = trainingVisible.outerProduct(hiddenPositive).subtract(negitive.outerProduct(hiddenNegitive));
        AGradient = trainingVisible.subtract(negitive);
        BGradient = hiddenPositive.subtract(hiddenNegitive);

        //rbm.W = rbm.W.add(gradient.scalarMultiply(learningRate));

        //rbm.a = rbm.a.add(trainingVisible.subtract(negitive).mapMultiplyToSelf(learningRate));
        //rbm.b = rbm.b.add(hiddenPositive.subtract(hiddenNegitive).mapMultiplyToSelf(learningRate));
    }


}
