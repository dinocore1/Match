package org.devsmart.match.rbm;


import com.google.common.base.Stopwatch;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class RBMTrainer {

    private static Logger logger = LoggerFactory.getLogger(RBMTrainer.class);

    private final RBM rbm;
    private final RBMMiniBatchCreator mMinibatchCreator;
    public Random random = new Random();

    private ExecutorService mExecutorService = Executors.newFixedThreadPool(1);
    public int numEpic = 1000;
    public int numGibbsSteps = 1;
    public double learningRate = 0.1;

    public RBMTrainer(RBM rbm, RBMMiniBatchCreator miniBatchCreator) {
        this.rbm = rbm;
        this.mMinibatchCreator = miniBatchCreator;
    }

    public void setNumTrainingThreads(int numThreads) {
        mExecutorService = Executors.newFixedThreadPool(numThreads);
    }


    public static void setInitialValues(RBM rbm, Collection<double[]> miniBatch, Random r) {

        //init weights zero-mean Gaussian with stddev of 0.01
        for(int i=0;i<rbm.visible.length;i++){
            for(int j=0;j<rbm.hidden.length;j++){
                rbm.W.setEntry(i, j, r.nextGaussian() * 0.01);
            }
        }

        rbm.b = new ArrayRealVector(rbm.hidden.length);

        SummaryStatistics[] trainStats = new SummaryStatistics[rbm.visible.length];
        for(int i=0;i<trainStats.length;i++){
            trainStats[i] = new SummaryStatistics();
        }

        for(double[] training : miniBatch){
            for(int i=0;i<rbm.visible.length;i++){
                trainStats[i].addValue(training[i]);
            }
        }

        /*
        for(int i=0;i<rbm.visible.length;i++){
            if(rbm.visible[i] instanceof GaussianNuron){
                double sigma = trainStats[i].getStandardDeviation();
                sigma = sigma == 0 ? 1 : sigma;
                ((GaussianNuron)rbm.visible[i]).setParams(0, 1);
                rbm.a.setEntry(i, trainStats[i].getMean());
            } else if(rbm.visible[i] instanceof BernoulliNuron){
                double mean = trainStats[i].getMean();
                if(mean == 0){
                    mean = Double.MIN_VALUE;
                }
                rbm.a.setEntry(i, FastMath.log(mean / (1-mean)));
            }
        }
        */

    }



    public void train() throws Exception {
        Stopwatch stopwatch = Stopwatch.createStarted();
        setInitialValues(rbm, mMinibatchCreator.createMiniBatch(), random);
        for(int i=0;i<numEpic;i++){
            final Collection<double[]> minibatch = mMinibatchCreator.createMiniBatch();
            ArrayList<Future<ContrastiveDivergence>> tasks = new ArrayList<Future<ContrastiveDivergence>>(minibatch.size());
            for(double[] trainingVisible : minibatch){
                tasks.add(mExecutorService.submit(new TrainingTask(trainingVisible)));
            }

            RealMatrix W = new Array2DRowRealMatrix(rbm.W.getRowDimension(), rbm.W.getColumnDimension());
            RealVector a = new ArrayRealVector(rbm.a.getDimension());
            RealVector b = new ArrayRealVector(rbm.b.getDimension());
            final double multiplier = learningRate / minibatch.size();
            for(Future<ContrastiveDivergence> task : tasks) {
                ContrastiveDivergence result = task.get();
                W = W.add(result.WGradient.scalarMultiply(multiplier));
                a = a.add(result.AGradient.mapMultiplyToSelf(multiplier));
                b = b.add(result.BGradient.mapMultiplyToSelf(multiplier));
            }


            rbm.W = rbm.W.add(W);
            rbm.a = rbm.a.add(a);
            rbm.b = rbm.b.add(b);

            if(i % 100 == 0) {
                //compute error using one of the minibatch examples
                SummaryStatistics errorStat = new SummaryStatistics();
                double[] trainingVisible = minibatch.iterator().next();
                double[] reconstruct = rbm.activateVisible(rbm.activateHidden(trainingVisible, random), random);
                for (int j = 0; j < reconstruct.length; j++) {
                    errorStat.addValue(trainingVisible[j] - reconstruct[j]);
                }
                final double error = FastMath.sqrt(errorStat.getSumsq() / errorStat.getN());
                logger.info("epoc: {} error: {}", i, error);
            }

        }
        stopwatch.stop();
        logger.info("Training took {}", stopwatch);

    }

    class TrainingTask implements Callable<ContrastiveDivergence> {

        private final ContrastiveDivergence retval = new ContrastiveDivergence();
        private final double[] input;

        TrainingTask(double[] input) {
            this.input = input;
        }

        @Override
        public ContrastiveDivergence call() throws Exception {
            retval.train(rbm, input, numGibbsSteps, random);
            return retval;
        }
    }
}
