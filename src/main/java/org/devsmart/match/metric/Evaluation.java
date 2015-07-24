package org.devsmart.match.metric;


import org.apache.commons.math3.stat.descriptive.moment.Mean;

import java.util.HashSet;

public class Evaluation<T extends Comparable<T>> {

    private ConfusionMatrix<T> mConfusionMatrix = new ConfusionMatrix<T>();
    //private Counter<T> mTruePositives = new Counter<T>();
    private Counter<T> mTrueNegitives = new Counter<T>();
    //private Counter<T> mFalseNegitives = new Counter<T>();
    //private Counter<T> mFalsePositives = new Counter<T>();
    private HashSet<T> mClassSet = new HashSet<T>();
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
        for(T clazz : mClassSet) {
            double p = getPrecision(clazz);
            if(!Double.isNaN(p)){
                mean.increment(p);
            }
        }
        return mean.getResult();
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
        for(T clazz : mClassSet) {
            double s = getSensitivity(clazz);
            if(!Double.isNaN(s)) {
                mean.increment(s);
            }
        }
        return mean.getResult();
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
        final double tn = getTrueNegitive(clazz);
        final double fp = getFalsePositives(clazz);

        return tn / (tn + fp);
    }

    public double getSpecificity() {
        Mean mean = new Mean();
        for(T clazz : mClassSet) {
            double s = getSpecificity(clazz);
            if(!Double.isNaN(s)){
                mean.increment(s);
            }
        }
        return mean.getResult();
    }

    public double getAccuracy(T clazz) {
        final double t = getTruePositive(clazz);
        final double population = mTotalCount;

        return  t / population;
    }

    public double getAccuracy() {
        final double tp = mConfusionMatrix.getTruePositives();
        return tp / mTotalCount;
    }

    public long getTruePositive(T clazz) {
        return mConfusionMatrix.getTruePositive(clazz);
        //return mTruePositives.getCount(clazz);
    }

    public long getTrueNegitive(T clazz) {
        return mTrueNegitives.getCount(clazz);
    }

    public long getFalsePositives(T clazz) {
        return mConfusionMatrix.getFalsePositive(clazz);
        //return mFalsePositives.getCount(clazz);
    }

    public long getFalseNegitive(T clazz) {
        return mConfusionMatrix.getFalsePositive(clazz);
        //return mFalseNegitives.getCount(clazz);
    }


    public double getF1Score() {
        return getFScore(1.0);
    }

    public double getFScore(double beta) {
        double precision = getPrecision();
        double recall = getSensitivity();

        double betaSquare = beta * beta;

        double retval = (1 + betaSquare) * ((precision*recall) / (betaSquare*precision+recall));
        return retval;
    }

    public void update(final T expectedValue, final T prediction) {

        mClassSet.add(expectedValue);
        mClassSet.add(prediction);
        mConfusionMatrix.update(expectedValue, prediction);


        if(!mTrueNegitives.getAllClasses().contains(expectedValue)) {
            mTrueNegitives.set(expectedValue, mTotalCount);
        }

        if(expectedValue.equals(prediction)) {
            //mTruePositives.increment(expectedValue);
            for(T clazz : mClassSet) {
                if(!clazz.equals(prediction)) {
                    mTrueNegitives.increment(clazz);
                }
            }
        } else {
            //mFalsePositives.increment(prediction);
            //mFalseNegitives.increment(expectedValue);
        }

        mTotalCount++;

    }

    public String getSummaryTxt() {
        StringBuilder buff = new StringBuilder();

        buff.append(String.format("F1 = %.6f\n", getF1Score()));
        buff.append(String.format("Acc = %.6f\n", getAccuracy()));
        buff.append('\n');

        for(T clazz : mClassSet) {
            buff.append(String.format("%s predicted %d times\n", clazz, mConfusionMatrix.getTruePositive(clazz)));
        }

        return buff.toString();
    }

    @Override
    public String toString() {
        return String.format("F1 = %.3f", getF1Score());
    }
}
