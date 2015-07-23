package org.devsmart.match.rbm;


import com.google.common.base.Stopwatch;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;
import org.devsmart.match.LossFunction;
import org.devsmart.match.WeightDecay;
import org.devsmart.match.neuralnet.MiniBatchCreator;
import org.devsmart.match.neuralnet.TraningData;
import org.devsmart.match.rbm.nuron.SigmoidNuron;
import org.devsmart.match.rbm.nuron.GaussianNuron;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class RBMTrainer {

    private static Logger logger = LoggerFactory.getLogger(RBMTrainer.class);

    public final RBM rbm;
    public final MiniBatchCreator mMinibatchCreator;
    public Random random = new Random();

    private ExecutorService mExecutorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    public LossFunction lossFunction = LossFunction.Square;
    public int numGibbsSteps = 1;
    public double learningRate = 0.1;
    public double momentum = 0.1;
    public double weightDecayCoefficient = 0.001;
    public WeightDecay weightDecayFunction = WeightDecay.L1;
    public List<IterationListener> iterationListeners = new ArrayList<IterationListener>();
    public Collection<TraningData> minibatch;

    public RBMTrainer(RBM rbm, MiniBatchCreator miniBatchCreator) {
        this.rbm = rbm;
        this.mMinibatchCreator = miniBatchCreator;
    }

    public void setNumTrainingThreads(int numThreads) {
        mExecutorService = Executors.newFixedThreadPool(numThreads);
    }


    public static void setInitialValues(RBM rbm, Collection<TraningData> miniBatch, Random r) {

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

        for(TraningData trainingData : miniBatch){
            for(int i=0;i<rbm.visible.length;i++){
                trainStats[i].addValue(trainingData.intput[i]);
            }
        }

        for(int i=0;i<rbm.visible.length;i++){
            if(rbm.visible[i] instanceof GaussianNuron){
                //double variance = trainStats[i].getStandardDeviation();
                rbm.a.setEntry(i, trainStats[i].getMean());
            } else if(rbm.visible[i] instanceof SigmoidNuron){
                double mean = trainStats[i].getMean();
                mean = Math.max(0.1, mean);
                mean = Math.min(0.9, mean);
                rbm.a.setEntry(i, FastMath.log(mean / (1-mean)));
            }
        }

    }


    public void train(Double sigmaErrorDiff, int errorWindow, Double maxError, long maxEpoch) throws Exception {
        Stopwatch stopwatch = Stopwatch.createStarted();
        setInitialValues(rbm, mMinibatchCreator.createMiniBatch(), random);

        RealMatrix lastW = new Array2DRowRealMatrix(rbm.W.getRowDimension(), rbm.W.getColumnDimension());
        RealVector lastA = new ArrayRealVector(rbm.a.getDimension());
        RealVector lastB = new ArrayRealVector(rbm.b.getDimension());

        for(int i=0;i<maxEpoch;i++){
            minibatch = mMinibatchCreator.createMiniBatch();
            ArrayList<Future<ContrastiveDivergence>> tasks = new ArrayList<Future<ContrastiveDivergence>>(minibatch.size());
            for(TraningData trainingVisible : minibatch){
                tasks.add(mExecutorService.submit(new TrainingTask(trainingVisible.intput)));
            }

            RealMatrix W = new Array2DRowRealMatrix(rbm.W.getRowDimension(), rbm.W.getColumnDimension());
            RealVector a = new ArrayRealVector(rbm.a.getDimension());
            RealVector b = new ArrayRealVector(rbm.b.getDimension());
            final double multiplier = 1.0 / minibatch.size();
            for(Future<ContrastiveDivergence> task : tasks) {
                ContrastiveDivergence result = task.get();
                W = W.add(result.WGradient.scalarMultiply(multiplier));
                a = a.add(result.AGradient.mapMultiplyToSelf(multiplier));
                b = b.add(result.BGradient.mapMultiplyToSelf(multiplier));
            }

            //W = W + l(g + m*g(t-1) - d*W)

            rbm.W = rbm.W.add( (W.add(lastW.scalarMultiply(momentum).subtract(weightDecayFunction.getWeightPenalty(rbm.W, weightDecayCoefficient)))).scalarMultiply(learningRate));
            rbm.a = rbm.a.add( (a.add(lastA.mapMultiply(momentum))) .mapMultiply(learningRate));
            rbm.b = rbm.b.add( (b.add(lastB.mapMultiply(momentum))) .mapMultiply(learningRate) );

            lastW = W;
            lastA = a;
            lastB = b;

            for(IterationListener listener : iterationListeners){
                listener.iterationDone(this, i);
            }

        }
        stopwatch.stop();
        logger.info("Training took {}", stopwatch);
    }

    public void train(double sigmaErrorDiff, long maxEpoc) throws Exception {
        train(sigmaErrorDiff, 10, null, maxEpoc);
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
