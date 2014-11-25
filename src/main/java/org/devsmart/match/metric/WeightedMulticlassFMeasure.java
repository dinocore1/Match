package org.devsmart.match.metric;


public class WeightedMulticlassFMeasure extends MulticlassFMeasure {

    public WeightedMulticlassFMeasure(float beta) {
        super(beta);
    }

    @Override
    public double getValue() {

        double total = 0;
        double num = 0;

        for(FMeasure f : labelMap.values()) {
            final double weight = f.actualPositive;
            num += weight * f.getValue();
            total += weight;
        }

        double retval = num / total;
        return retval;
    }
}
