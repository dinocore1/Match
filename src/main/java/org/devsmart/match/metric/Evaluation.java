package org.devsmart.match.metric;


public class Evaluation<T> {

    private Counter<T> mTruePositives = new Counter<T>();
    private Counter<T> mTrueNegitives = new Counter<T>();
    private Counter<T> mFalseNegitives = new Counter<T>();
    private Counter<T> mFalsePositives = new Counter<T>();
    private long mTotalCount = 0;


    /**
     * Precision is the ratio of true positives to all predicted positives
     *
     * @return precision
     */
    public double getPrecision() {
        final double tp = getTruePositive();
        final double fp = getFalsePositives();

        final double pt = tp + fp;

        if(pt == 0) {
            return 0;
        } else {
            return tp / pt;
        }
    }

    /**
     *  Sensitivity, (a.k.a recall or true positive rate) measures the proportion
     *  of positives that are correctly identified as such (e.g., the percentage
     *  of sick people who are correctly identified as having the condition).
     *
     * @return sensitivity
     */
    public double getSensitivity() {
        final double tp = getTruePositive();
        final double fn = getFalseNegitive();

        final double t = tp + fn;
        if(t == 0) {
            return 0;
        } else {
            return tp / t;
        }
    }

    /**
     * Specificity (also called the true negative rate) measures the proportion of
     * negatives that are correctly identified as such (e.g., the percentage of
     * healthy people who are correctly identified as not having the condition).
     * @return
     */
    public double getSpecificity() {
        final double tn = getTrueNegitive();
        final double fp = getFalsePositives();

        final double t = tn + fp;
        if(t == 0) {
            return 0;
        } else {
            return tn / t;
        }
    }

    public double getAccuracy() {
        if(mTotalCount == 0) {
            return 0;
        } else {
            final double tp = getTruePositive();

            return  tp / mTotalCount;
        }
    }

    public long getTruePositive() {
        return mTruePositives.getTotal();
    }

    public long getTrueNegitive() {
        return mTrueNegitives.getTotal();
    }

    public long getFalsePositives() {
        return mFalsePositives.getTotal();
    }

    public long getFalseNegitive() {
        return mFalseNegitives.getTotal();
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

        if(expectedValue.equals(prediction)) {
            mTruePositives.increment(prediction);
            for(T clazz : mTruePositives.getAllClasses()) {
                if(!clazz.equals(expectedValue)) {
                    mTrueNegitives.increment(clazz);
                }
            }
        } else {
            mFalsePositives.increment(prediction);
            mFalseNegitives.increment(expectedValue);
        }

        mTotalCount++;

    }

    public String getSummaryTxt() {
        StringBuilder buff = new StringBuilder();

        buff.append(String.format("F1 = %.6f\n", getF1Score()));
        buff.append(String.format("Acc = %.6f\n", getAccuracy()));
        buff.append('\n');

        for(T clazz : mTruePositives.getAllClasses()) {
            buff.append(String.format("%s predicted %d times\n", clazz, mTruePositives.getCount(clazz)));
        }

        return buff.toString();
    }

    @Override
    public String toString() {
        return String.format("F1 = %.3f", getF1Score());
    }

}
