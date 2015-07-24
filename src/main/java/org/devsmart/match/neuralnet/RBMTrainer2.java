package org.devsmart.match.neuralnet;


import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;
import org.devsmart.match.metric.Evaluation;
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
        this.weightUpdate = new WeightUpdate(rbm, 0.1f);
        this.contrastiveDivergence = new ContrastiveDivergence2(rbm);
        this.miniBatchCreator = miniBatchCreator;

    }

    public void setLearningRate(float learningRate) {
        weightUpdate.setLearningrate(learningRate);
    }

    public void setMomentum(float momentum) {
        weightUpdate.setMomentum(momentum);
    }

    public static void setInitialValues(RBM2 rbm, MiniBatchCreator miniBatchCreator, Random r) {

        float[] weights = new float[rbm.numHidden*rbm.numVisible+rbm.numHidden+rbm.numVisible];

        //init weights zero-mean Gaussian with stddev of 0.01
        for(int i=0;i<rbm.numVisible*rbm.numHidden;i++){
            weights[i] = (float)r.nextGaussian() * 0.01f;
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

        for(int i=0;i<rbm.numHidden;i++){
            //weights[rbm.getHiddenBias(i)] = -4;
        }

        for(int i=0;i<rbm.numVisible;i++){
            double mean = trainStats[i].getMean();
            mean = Math.max(0.1, mean);
            mean = Math.min(0.9, mean);
            float value = (float) FastMath.log(mean / (1 - mean));
            weights[rbm.getVisibleBias(i)] = value;
        }

        rbm.setWeights(weights);
    }

    public static void calcF1(Evaluation measure, double[] inputVisible, float[] reconstructVisible) {
        assert inputVisible.length == reconstructVisible.length;
        for(int i=0;i<inputVisible.length;i++){
            measure.update(Math.round(inputVisible[i]) == 1, Math.round(reconstructVisible[i]) == 1);
        }
    }

    public void calcError(long epoc, RBM2 rbm, Collection<TraningData> minibatch) {

        Evaluation f1 = new Evaluation();
        SummaryStatistics error = new SummaryStatistics();

        for(TraningData data : minibatch) {

            float[] values = rbm.activateHidden(data.intput);
            values = rbm.activateVisible(values);

            calcF1(f1, data.intput, values);

            for(int i=0;i<values.length;i++){
                error.addValue(Math.abs(data.intput[i] - values[i]));
            }
        }


        double errorValue = error.getMean();

        logger.info("epoch {} error: {} f1: {}", epoc, errorValue, f1.getF1Score());
    }

    public void train(final long maxEpoch) {

        setInitialValues(rbm, miniBatchCreator, random);

        calcError(0, rbm, miniBatchCreator.createMiniBatch());
        for(long i=0;i<maxEpoch;i++) {

            final Collection<TraningData> minibatch = miniBatchCreator.createMiniBatch();
            contrastiveDivergence.resetGradient();

            for(TraningData data : minibatch) {
                contrastiveDivergence.doIt(data.intput, numGibbsSamples, random);
            }

            float[] gradient = contrastiveDivergence.getGradient();
            weightUpdate.update(gradient);


            calcError(i+1, rbm, minibatch);

        }

    }
}
