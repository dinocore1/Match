package org.devsmart.match.metric;


import java.util.HashMap;

public abstract class MulticlassFMeasure {

    public final float beta;
    protected HashMap<String, FMeasure> labelMap = new HashMap<String, FMeasure>();

    public MulticlassFMeasure(float beta) {
        this.beta = beta;
    }

    public void update(String knownLabel, String predictedLabel) {
        FMeasure fmeasure = labelMap.get(knownLabel);
        if(fmeasure == null) {
            fmeasure = new FMeasure(beta);
            labelMap.put(knownLabel, fmeasure);
        }
        fmeasure.update(true, knownLabel.equals(predictedLabel));
    }

    public abstract double getValue();

}
