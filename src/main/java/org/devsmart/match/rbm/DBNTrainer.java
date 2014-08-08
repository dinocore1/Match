package org.devsmart.match.rbm;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class DBNTrainer {

    private ExecutorService mExecutorService = Executors.newFixedThreadPool(4);
    private final MiniBatchCreator mMinibachCreator;
    private final DBN dbn;
    public int numGibbsSteps = 1;
    public double learningRate = 0.1;
    public int numEpocsPerLayer = 4000;
    public Random random = new Random();

    public DBNTrainer(DBN dbn, MiniBatchCreator callback){
        this.dbn = dbn;
        this.mMinibachCreator = callback;
    }

    public void train() throws Exception {
        for(int layer=0;layer<dbn.rbms.size();layer++){
            RBM rbm = dbn.rbms.get(layer);
            RBMTrainer.setInitialValues(rbm, mMinibachCreator.createMiniBatch().iterator(), random);
            for(int i=0;i<numEpocsPerLayer;i++){

                Collection<RealVector> miniBatch = mMinibachCreator.createMiniBatch();
                ArrayList<Future<ContrastiveDivergence>> tasks = new ArrayList<Future<ContrastiveDivergence>>(miniBatch.size());
                for(RealVector input : mMinibachCreator.createMiniBatch()) {
                    tasks.add(mExecutorService.submit(new TrainingTask(input, layer)));
                }
                RealMatrix W = new Array2DRowRealMatrix(rbm.W.getRowDimension(), rbm.W.getColumnDimension());
                RealVector a = new ArrayRealVector(rbm.a.getDimension());
                RealVector b = new ArrayRealVector(rbm.b.getDimension());
                double lr = learningRate / miniBatch.size();
                for(Future<ContrastiveDivergence> task : tasks) {
                    ContrastiveDivergence result = task.get();
                    W = W.add(result.WGradient.scalarMultiply(lr));
                    a = a.add(result.AGradient.mapMultiplyToSelf(lr));
                    b = b.add(result.BGradient.mapMultiplyToSelf(lr));
                }

                rbm.W = rbm.W.add(W);
                rbm.a = rbm.a.add(a);
                rbm.b = rbm.b.add(b);

                {
                    //compute error using one of the minibatch examples
                    SummaryStatistics errorStat = new SummaryStatistics();
                    RealVector trainingVisible = miniBatch.iterator().next();
                    RealVector reconstruct = dbn.propagateDown(dbn.propagateUp(trainingVisible, layer+1, random), layer, random);
                    for (int j = 0; j < reconstruct.getDimension(); j++) {
                        errorStat.addValue(trainingVisible.getEntry(j) - reconstruct.getEntry(j));
                    }
                    final double error = FastMath.sqrt(errorStat.getSumsq() / errorStat.getN());
                    System.out.println(String.format("layer: %d epoc: %d error: %.5g", layer, i, error));
                }
            }
        }
    }

    private class TrainingTask implements Callable<ContrastiveDivergence> {

        private final ContrastiveDivergence retval = new ContrastiveDivergence();
        private final RealVector input;
        private final int layer;

        public TrainingTask(RealVector input, int layer){
            this.input = input;
            this.layer = layer;
        }

        @Override
        public ContrastiveDivergence call() throws Exception {
            retval.train(dbn.rbms.get(layer), dbn.propagateUp(input, layer, random), numGibbsSteps, random);
            return retval;
        }
    }


}
