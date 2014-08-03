package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

public class RBMTrainer {

    private final RBM rbm;
    public int numGibbsSteps = 1;
    public double learningRate = 0.1;

    public RBMTrainer(RBM rbm) {
        this.rbm = rbm;
    }


    public void setInitialValues(Iterator<RealVector> miniBatch) {

        //init weights zero-mean Gaussian with stddev of 0.01
        for(int i=0;i<rbm.numVisible;i++){
            for(int j=0;j<rbm.numHidden;j++){
                rbm.W.setEntry(i, j, rbm.r.nextGaussian() * 0.01);
            }
        }

        rbm.b = new ArrayRealVector(rbm.numHidden);

        double[] means = new double[rbm.numVisible];
        int n = 0;
        while(miniBatch.hasNext()){
            n++;
            RealVector training = miniBatch.next();
            for(int i=0;i<rbm.numVisible;i++){
                means[i] += (training.getEntry(i) - means[i]) / n;
            }

        }
        if(rbm.isGaussianVisible){
           rbm.a = new ArrayRealVector(means);
        } else {
            rbm.a = new ArrayRealVector(rbm.numVisible);
            for(int i=0;i<rbm.numVisible;i++){
                rbm.a.setEntry(i, FastMath.log(means[i] / (1-means[i])));
            }
        }
    }


    public void train(Collection<RealVector> trainingData, int numEpic) {
        for(int i=0;i<numEpic;i++){
            for(RealVector trainingVisible : trainingData){
                RealVector negitive = gibbsSample(trainingVisible, numGibbsSteps);

                RealVector hiddenPositive = rbm.activateHidden(trainingVisible);
                RealVector hiddenNegitive = rbm.activateHidden(negitive);

                RealMatrix gradient = trainingVisible.outerProduct(hiddenPositive).subtract(negitive.outerProduct(hiddenNegitive));
                rbm.W = rbm.W.add(gradient.scalarMultiply(learningRate));

                rbm.a = rbm.a.add(trainingVisible.subtract(negitive).mapMultiplyToSelf(learningRate));
                rbm.b = rbm.b.add(hiddenPositive.subtract(hiddenNegitive).mapMultiplyToSelf(learningRate));


            }
        }
    }

    public RealVector gibbsSample(RealVector input, int numSteps) {
        for(int k=0;k<numSteps;k++) {
            RealVector hidden = rbm.activateHidden(input);
            input = rbm.activateVisible(hidden);
        }
        return input;
    }
}
