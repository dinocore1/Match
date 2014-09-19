package org.devsmart.match.rbm;

import org.devsmart.match.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DBNTrainer {

    private static Logger logger = LoggerFactory.getLogger(DBNTrainer.class);

    public ExecutorService mExecutorService = Executors.newFixedThreadPool(4);
    private final DBNMiniBatchCreator mMinibachCreator;
    private final DBN dbn;
    public int numGibbsSteps = 1;
    public double learningRate = 0.1;
    public LossFunction lossFunction = LossFunction.SumOfSquares;
    public Random random = new Random();
    public double momentum = 0.9;
    public double weightDecayCoefficient = 0.0001;

    public DBNTrainer(DBN dbn, DBNMiniBatchCreator callback){
        this.dbn = dbn;
        this.mMinibachCreator = callback;
    }

    public void train(Double sigmaErrorDiff, int errorWindow, Double maxError, long maxEpoch) throws Exception {
        for(int layer=0;layer<dbn.mRBMs.size();layer++){
            logger.info("training layer: {}", layer);
            RBM rbm = dbn.mRBMs.get(layer);

            final int currentLayer = layer;
            RBMTrainer trainer = new RBMTrainer(rbm, new RBMMiniBatchCreator() {
                @Override
                public Collection<double[]> createMiniBatch() {
                    Collection<double[][]> dbnminibatch = mMinibachCreator.createMinibatch();
                    ArrayList<double[]> minibatch = new ArrayList<double[]>(dbnminibatch.size());
                    for(double[][] trainingdata : dbnminibatch) {
                        minibatch.add(dbn.propagateUp(currentLayer, trainingdata, random));
                    }
                    return minibatch;
                }
            });
            trainer.numGibbsSteps = numGibbsSteps;
            trainer.learningRate = learningRate;
            trainer.lossFunction = lossFunction;
            trainer.weightDecayCoefficient = weightDecayCoefficient;
            trainer.momentum = momentum;
            trainer.train(sigmaErrorDiff, errorWindow, maxError, maxEpoch);
        }
    }

    public void train(double sigmaErrorDiff, long maxEpoc) throws Exception {
        train(sigmaErrorDiff, 10, null, maxEpoc);
    }


}
