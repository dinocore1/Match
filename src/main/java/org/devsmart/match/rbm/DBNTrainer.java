package org.devsmart.match.rbm;

import org.apache.commons.math3.linear.RealVector;

import java.util.Collection;

public class DBNTrainer {

    private final DBN dbn;
    int numGibbsSteps = 1;
    double learningRate = 0.01;

    public DBNTrainer(DBN dbn){
        this.dbn = dbn;
    }

    public void train(Collection<RealVector> trainingData, int numEpocs) {
        for(int layer=0;layer<dbn.rbms.size();layer++){
            RBM rbm = dbn.rbms.get(layer);
            RBMTrainer.setInitialValues(rbm, trainingData.iterator());
            for(int i=0;i<numEpocs;i++){
                for(RealVector input : trainingData) {
                    input = dbn.propagateUp(input, layer);
                    ContrastiveDivergence.train(rbm, input, numGibbsSteps, learningRate);
                }
            }
        }
    }


}
