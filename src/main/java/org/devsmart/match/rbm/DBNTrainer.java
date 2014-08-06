package org.devsmart.match.rbm;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class DBNTrainer {

    public interface Callback {
        Collection<RealVector> createMiniBatch();
    }

    private ExecutorService mExecutorService = Executors.newFixedThreadPool(4);
    private final Callback mCallback;
    private final DBN dbn;
    int numGibbsSteps = 1;
    double learningRate = 0.01;
    int numEpocsPerLayer = 1000;

    public DBNTrainer(DBN dbn, Callback callback){
        this.dbn = dbn;
        this.mCallback = callback;
    }

    public void train() throws Exception {
        for(int layer=0;layer<dbn.rbms.size();layer++){
            RBM rbm = dbn.rbms.get(layer);
            RBMTrainer.setInitialValues(rbm, mCallback.createMiniBatch().iterator());
            for(int i=0;i<numEpocsPerLayer;i++){

                Collection<RealVector> miniBatch = mCallback.createMiniBatch();
                ArrayList<Future<ContrastiveDivergence>> tasks = new ArrayList<Future<ContrastiveDivergence>>(miniBatch.size());
                for(RealVector input : mCallback.createMiniBatch()) {
                    tasks.add(mExecutorService.submit(new TrainingTask(input, layer)));
                }
                RealMatrix W = new Array2DRowRealMatrix(rbm.W.getRowDimension(), rbm.W.getColumnDimension());
                RealVector a = new ArrayRealVector(rbm.a.getDimension());
                RealVector b = new ArrayRealVector(rbm.b.getDimension());
                for(Future<ContrastiveDivergence> task : tasks) {
                    ContrastiveDivergence result = task.get();
                    W = W.add(result.WGradient);
                    a = a.add(result.AGradient);
                    b = b.add(result.BGradient);
                }

                final double multiplier = learningRate/miniBatch.size();
                //final double multiplier = 1.0;
                rbm.W = rbm.W.add(W.scalarMultiply(multiplier));
                rbm.a = rbm.a.add(a.mapMultiplyToSelf(multiplier));
                rbm.b = rbm.b.add(b.mapMultiplyToSelf(multiplier));

                {
                    //compute error using one of the minibatch examples
                    SummaryStatistics errorStat = new SummaryStatistics();
                    RealVector trainingVisible = miniBatch.iterator().next();
                    RealVector reconstruct = dbn.propagateDown(dbn.propagateUp(trainingVisible, layer+1), layer, layer);
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
            retval.train(dbn.rbms.get(layer), dbn.propagateUp(input, layer), numGibbsSteps);
            return retval;
        }
    }


}
