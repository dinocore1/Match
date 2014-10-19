package org.devsmart.match.neuralnet;


import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Random;

public class RBMTrainer2 {

    Logger logger = LoggerFactory.getLogger(RBMTrainer2.class);

    private final RBM2 rbm;
    private final MiniBatchCreator miniBatchCreator;
    private final ContrastiveDivergence2 contrastiveDivergence;

    public int numGibbsSamples = 1;
    private WeightUpdate weightUpdate;
    public Random random = new Random();

    public RBMTrainer2(RBM2 rbm, MiniBatchCreator miniBatchCreator) {
        this.rbm = rbm;
        this.weightUpdate = new WeightUpdate(rbm.numVisible*rbm.numHidden+rbm.numVisible+rbm.numHidden, 0.1f);
        this.contrastiveDivergence = new ContrastiveDivergence2(rbm);
        this.miniBatchCreator = miniBatchCreator;

    }

    public static void setInitialValues(RBM2 rbm, MiniBatchCreator miniBatchCreator, Random r) {
        //init weights zero-mean Gaussian with stddev of 0.01
        for(int i=0;i<rbm.numVisible*rbm.numHidden;i++){
            rbm.weights[i] = (float)r.nextGaussian() * 0.01f;
        }

        SummaryStatistics[] trainStats = new SummaryStatistics[rbm.numVisible];
        for(int i=0;i<rbm.numVisible;i++){
            trainStats[i] = new SummaryStatistics();
        }

        for(TraningData training : miniBatchCreator.createMiniBatch()){
            for(int i=0;i<rbm.numVisible;i++){
                trainStats[i].addValue(training.intput[i]);
            }
        }

        for(int i=0;i<rbm.numVisible;i++){
            double mean = trainStats[i].getMean();
            mean = Math.max(0.0001, mean);
            mean = Math.min(0.9999, mean);
            float value = (float) FastMath.log(mean / (1 - mean));
            rbm.weights[rbm.numVisible*rbm.numHidden+rbm.numHidden+i] = value;
        }
    }

    public static double calcError(RBM2 rbm, Collection<TraningData> minibatch) {

        SummaryStatistics error = new SummaryStatistics();

        for(TraningData data : minibatch) {

            float[] values = rbm.activateHidden(data.intput);
            values = rbm.activateVisible(values);

            for(int i=0;i<values.length;i++){
                error.addValue(Math.abs(data.intput[i] - values[i]));
            }
        }

        return error.getMean();
    }

    public void train(final long maxEpoch) {

        setInitialValues(rbm, miniBatchCreator, random);

        for(long i=0;i<maxEpoch;i++) {

            contrastiveDivergence.resetGradient();
            final Collection<TraningData> minibatch = miniBatchCreator.createMiniBatch();

            for(TraningData data : minibatch) {
                contrastiveDivergence.doIt(data.intput, numGibbsSamples, random);
            }

            float[] gradient = contrastiveDivergence.getGradient();
            float[] weights = weightUpdate.update(gradient);
            rbm.setWeights(weights);

            double error = calcError(rbm, minibatch);
            logger.info("epoch {} error: {}", i, error);

        }

    }
}
