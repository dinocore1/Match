package org.devsmart.match.rbm;


import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.devsmart.match.neuralnet.TraningData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ScoreIterationListener implements IterationListener {

    private static final Logger logger = LoggerFactory.getLogger(ScoreIterationListener.class);

    private final int mFrequency;

    public ScoreIterationListener() {
        mFrequency = 10;
    }

    public ScoreIterationListener(int frequency) {
        mFrequency = frequency;
    }

    @Override
    public void iterationDone(RBMTrainer trainer, int iteration) {

        if(iteration % mFrequency == 0) {
            SummaryStatistics lossStats = new SummaryStatistics();
            for (TraningData data : trainer.minibatch) {
                double[] reconstruct = trainer.rbm.activateVisible(trainer.rbm.activateHidden(data.intput));
                double loss = trainer.lossFunction.loss(data.intput, reconstruct);
                lossStats.addValue(loss);
            }

            final double stderr = lossStats.getStandardDeviation() / Math.sqrt(lossStats.getN());
            logger.info(String.format("%d : avg loss: %g +/- %g", iteration, lossStats.getMean(), 1.96 * stderr));
        }

    }
}
