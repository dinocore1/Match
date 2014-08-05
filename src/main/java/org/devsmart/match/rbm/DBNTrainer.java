package org.devsmart.match.rbm;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import java.util.Collection;

public class DBNTrainer {

    public interface Callback {
        Collection<RealVector> createMiniBatch();
    }

    private final Callback mCallback;
    private final DBN dbn;
    int numGibbsSteps = 1;
    double learningRate = 0.01;
    int numEpocsPerLayer = 1000;

    public DBNTrainer(DBN dbn, Callback callback){
        this.dbn = dbn;
        this.mCallback = callback;
    }

    public void train() {
        for(int layer=0;layer<dbn.rbms.size();layer++){
            RBM rbm = dbn.rbms.get(layer);
            RBMTrainer.setInitialValues(rbm, mCallback.createMiniBatch().iterator());
            for(int i=0;i<numEpocsPerLayer;i++){

                SummaryStatistics avgError = new SummaryStatistics();
                for(RealVector input : mCallback.createMiniBatch()) {
                    input = dbn.propagateUp(input, layer);
                    ContrastiveDivergence.train(rbm, input, numGibbsSteps, learningRate);

                    //compute error
                    SummaryStatistics errorStat = new SummaryStatistics();
                    RealVector reconstruct = rbm.activateVisible(rbm.activateHidden(input));
                    for(int j=0;j<reconstruct.getDimension();j++){
                        errorStat.addValue(input.getEntry(j)-reconstruct.getEntry(j));
                    }
                    final double error = FastMath.sqrt(errorStat.getSumsq() / errorStat.getN());
                    avgError.addValue(error);
                }
                System.out.println(String.format("layer: %d epoc: %d error: %.5g", layer, i, avgError.getMean()));
            }
        }
    }


}
