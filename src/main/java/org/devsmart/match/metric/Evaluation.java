package org.devsmart.match.metric;


import org.apache.commons.math3.stat.descriptive.moment.Mean;

public class Evaluation<T extends Comparable<T>> {

    private ConfusionMatrix<T> mConfusionMatrix = new ConfusionMatrix<T>();
    private long mTotalCount = 0;


    /**
     * Precision is the ratio of true positives to all predicted positives
     *
     * @return precision
     */
    public double getPrecision(T clazz) {
        final double tp = mConfusionMatrix.getTruePositive(clazz);
        final double fp = mConfusionMatrix.getFalsePositive(clazz);

        return tp / (tp + fp);
    }

    /**
     * Precision is the ratio of true positives to all predicted positives
     *
     * @return precision
     */
    public double getPrecision() {
        Mean mean = new Mean();
        for(T c : mConfusionMatrix.getClassSet()) {
            double p = getPrecision(c);
            if(!Double.isNaN(p)) {
                mean.increment(p);
            }
        }


        return mean.getResult();

        /*
        final double tp = mConfusionMatrix.getTruePositive();
        final double fp = mConfusionMatrix.getFalsePositive();

        return tp / (tp + fp);
        */
    }

    /**
     *  Sensitivity, (a.k.a recall or true positive rate) measures the proportion
     *  of positives that are correctly identified as such (e.g., the percentage
     *  of sick people who are correctly identified as having the condition).
     *
     * @return sensitivity
     */
    public double getSensitivity(T clazz) {
        final double tp = mConfusionMatrix.getTruePositive(clazz);
        final double fn = mConfusionMatrix.getFalseNegitive(clazz);

        return tp / (tp + fn);
    }

    public double getRecall(T clazz) {
        return getSensitivity(clazz);
    }

    /**
     *  Sensitivity, (a.k.a recall or true positive rate) measures the proportion
     *  of positives that are correctly identified as such (e.g., the percentage
     *  of sick people who are correctly identified as having the condition).
     *
     * @return sensitivity
     */
    public double getSensitivity() {

        Mean mean = new Mean();
        for(T c : mConfusionMatrix.getClassSet()) {
            double s = getSensitivity(c);
            if(!Double.isNaN(s)) {
                mean.increment(s);
            }
        }

        return mean.getResult();

        /*
        final double tp = mConfusionMatrix.getTruePositive();
        final double fn = mConfusionMatrix.getFalseNegitive();

        return tp / (tp + fn);
        */
    }

    public double getRecall() {
        return getSensitivity();
    }

    /**
     * Specificity (also called the true negative rate) measures the proportion of
     * negatives that are correctly identified as such (e.g., the percentage of
     * healthy people who are correctly identified as not having the condition).
     * @return
     */
    public double getSpecificity(T clazz) {
        final double tn = mConfusionMatrix.getTrueNegitive(clazz);
        final double fp = mConfusionMatrix.getFalsePositive(clazz);

        return tn / (tn + fp);
    }

    public double getSpecificity() {
        final double tn = mConfusionMatrix.getTrueNegitive();
        final double fp = mConfusionMatrix.getFalsePositive();

        return tn / (tn + fp);
    }

    public double getAccuracy(T clazz) {
        final double t = getTruePositive(clazz);
        final double population = mTotalCount;

        return t / population;
    }

    public double getAccuracy() {
        final double tp = mConfusionMatrix.getTruePositive();
        return tp / mTotalCount;
    }

    public long getTruePositive(T clazz) {
        return mConfusionMatrix.getTruePositive(clazz);
    }

    public long getTrueNegitive(T clazz) {
        return mConfusionMatrix.getTrueNegitive(clazz);
    }

    public long getFalsePositives(T clazz) {
        return mConfusionMatrix.getFalsePositive(clazz);
    }

    public long getFalseNegitive(T clazz) {
        return mConfusionMatrix.getFalsePositive(clazz);
    }


    public double getF1Score() {
        return getFScore(1.0);
    }

    public double getFScore(double beta, T clazz) {
        double precision = getPrecision(clazz);
        double recall = getSensitivity(clazz);

        double betaSquare = beta * beta;

        double retval = (1 + betaSquare) * ((precision*recall) / (betaSquare*precision+recall));
        return retval;
    }

    public double getFScore(double beta) {

        double precision = getPrecision();
        double recall = getSensitivity();

        double betaSquare = beta * beta;

        double retval = (1 + betaSquare) * ((precision*recall) / (betaSquare*precision+recall));
        return retval;

        /*
        Mean mean = new Mean();
        for(T c : mConfusionMatrix.getClassSet()) {
            double f1 = getFScore(beta, c);
            if(!Double.isNaN(f1)) {
                mean.increment(f1);
            }
        }
        return mean.getResult();
        */
    }

    public void update(final T expectedValue, final T prediction) {

        mConfusionMatrix.update(expectedValue, prediction);
        mTotalCount++;

    }

    public String getSummaryTxt() {
        StringBuilder buff = new StringBuilder();

        buff.append(String.format("F1 = %.6f\n", getF1Score()));
        buff.append(String.format("Acc = %.6f\n", getAccuracy()));
        buff.append('\n');

        for(T clazz : mConfusionMatrix.getClassSet()) {
            buff.append(String.format("%s predicted %d times\n", clazz, mConfusionMatrix.getTruePositive(clazz)));
        }

        return buff.toString();
    }

    @Override
    public String toString() {
        return String.format("F1 = %.3f", getF1Score());
    }
}
