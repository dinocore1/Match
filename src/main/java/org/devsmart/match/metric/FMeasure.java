package org.devsmart.match.metric;


public class FMeasure {

    public static FMeasure F1() {
        return new FMeasure(1.0);
    }

    public final double beta;
    int truePositive = 0;
    int predictedPositive = 0;
    int actualPositive = 0;

    public FMeasure(double beta) {
        this.beta = beta;
    }

    /**
     * Precision is the ratio of true positives to all predicted positives
     *
     * @return precision
     */
    public double getPrecisionScore() {
        return predictedPositive > 0 ? (double) truePositive / (double) predictedPositive : 0;
    }

    /**
     *  Recall is the ratio of true positives to all actual positives
     *
     * @return recall
     */
    public double getRecallScore() {
        return actualPositive > 0 ? (double) truePositive / (double) actualPositive : 0;
    }

    public double getValue() {
        double precision = getPrecisionScore();
        double recall = getRecallScore();
        double betaSquared = beta * beta;

        return (1+betaSquared) * ((precision * recall) / (betaSquared*precision + recall));
    }

    public void update(final boolean knownStandard, final boolean prediction) {
        actualPositive += knownStandard ? 1 : 0;
        predictedPositive += prediction ? 1 : 0;
        truePositive += prediction && knownStandard ? 1 : 0;
    }

    @Override
    public String toString() {
        return Double.toString(getValue());
    }

}
