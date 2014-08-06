package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class RBMTrainer {

    private final RBM rbm;
    private final MiniBatchCreator mMinibatchCreator;

    private ExecutorService mExecutorService = Executors.newFixedThreadPool(4);
    public int numEpic = 1000;
    public int numGibbsSteps = 1;
    public double learningRate = 0.2;

    public RBMTrainer(RBM rbm, MiniBatchCreator miniBatchCreator) {
        this.rbm = rbm;
        this.mMinibatchCreator = miniBatchCreator;
    }


    public static void setInitialValues(RBM rbm, Iterator<RealVector> miniBatch) {

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
                if(means[i] == 0){
                    means[i] = Double.MIN_VALUE;
                }
                rbm.a.setEntry(i, FastMath.log(means[i] / (1-means[i])));
            }
        }
    }



    public void train() throws Exception {
        setInitialValues(rbm, mMinibatchCreator.createMiniBatch().iterator());
        for(int i=0;i<numEpic;i++){
            final Collection<RealVector> minibatch = mMinibatchCreator.createMiniBatch();
            ArrayList<Future<ContrastiveDivergence>> tasks = new ArrayList<Future<ContrastiveDivergence>>(minibatch.size());
            for(RealVector trainingVisible : minibatch){
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


        }
    }

    class TrainingTask implements Callable<ContrastiveDivergence> {

        private final ContrastiveDivergence retval = new ContrastiveDivergence();
        private final RealVector input;

        TrainingTask(RealVector input) {
            this.input = input;
        }

        @Override
        public ContrastiveDivergence call() throws Exception {
            retval.train(rbm, input, numGibbsSteps);
            return retval;
        }
    }
}
