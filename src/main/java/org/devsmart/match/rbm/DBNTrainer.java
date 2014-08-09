package org.devsmart.match.rbm;

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
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class DBNTrainer {

    private static Logger logger = LoggerFactory.getLogger(DBNTrainer.class);

    private ExecutorService mExecutorService = Executors.newFixedThreadPool(4);
    private final DBNMiniBatchCreator mMinibachCreator;
    private final DBN dbn;
    public int numGibbsSteps = 1;
    public double learningRate = 0.1;
    public int numEpocsPerLayer = 4000;
    public Random random = new Random();

    public DBNTrainer(DBN dbn, DBNMiniBatchCreator callback){
        this.dbn = dbn;
        this.mMinibachCreator = callback;
    }

    private void setInitialValuesForLayer(int layer) {
        Collection<double[][][]> minibatch = mMinibachCreator.createMinibatch();

        Collection<double[]> rbmMinibatch = new ArrayList<double[]>(minibatch.size());
        for(double[][][] input : minibatch) {
            double[] visible = dbn.propagateUp(layer, input, random);
            rbmMinibatch.add(dbn.getVisibleForLayer(layer, input, visible));
        }

        RBMTrainer.setInitialValues(dbn.mRBMs.get(layer), rbmMinibatch, random);
    }

    public void train() throws Exception {
        for(int layer=0;layer<dbn.mRBMs.size();layer++){
            RBM rbm = dbn.mRBMs.get(layer);
            setInitialValuesForLayer(layer);
            for(int i=0;i<numEpocsPerLayer;i++){

                Collection<double[][][]> miniBatch = mMinibachCreator.createMinibatch();
                ArrayList<Future<ContrastiveDivergence>> tasks = new ArrayList<Future<ContrastiveDivergence>>(miniBatch.size());
                for(double[][][] input : miniBatch) {
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


                if(i % 100 == 0) {
                    //compute error using one of the minibatch examples
                    SummaryStatistics errorStat = new SummaryStatistics();
                    double[][][] input = miniBatch.iterator().next();

                    double[] layerInput = dbn.propagateUp(layer, input, random);
                    layerInput = dbn.getVisibleForLayer(layer, input, layerInput);

                    double[] reconstruct = rbm.getVisibleInput(rbm.activateHidden(layerInput, random));
                    for (int j = 0; j < reconstruct.length; j++) {
                        errorStat.addValue(layerInput[j] - reconstruct[j]);
                    }
                    final double error = FastMath.sqrt(errorStat.getSumsq() / errorStat.getN());
                    logger.info("layer: {} epoc: {} error: {}", layer, i, error);
                }
            }
        }
    }

    private class TrainingTask implements Callable<ContrastiveDivergence> {

        private final ContrastiveDivergence retval = new ContrastiveDivergence();
        private final double[][][] input;
        private final int layer;

        public TrainingTask(double[][][] input, int layer){
            this.input = input;
            this.layer = layer;
        }

        @Override
        public ContrastiveDivergence call() throws Exception {
            double[] layerInput = dbn.propagateUp(layer, input, random);
            layerInput = dbn.getVisibleForLayer(layer, input, layerInput);
            retval.train(dbn.mRBMs.get(layer), layerInput, numGibbsSteps, random);
            return retval;
        }
    }


}
