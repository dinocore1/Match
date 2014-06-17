package org.devsmart.match;

import au.com.bytecode.opencsv.CSVReader;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.LinkedList;

public class CartTest {

    @Test
    public void testCart() throws IOException {

        CSVReader reader = new CSVReader(new InputStreamReader(getClass().getResourceAsStream("ex0.txt")), '\t');

        LinkedList<PointValuePair> data = new LinkedList<PointValuePair>();
        String[] line = null;
        while((line = reader.readNext()) != null){
            double[] feature = new double[2];
            feature[0] = Double.parseDouble(line[0]);
            feature[1] = Double.parseDouble(line[1]);
            PointValuePair p = new PointValuePair(feature, Double.parseDouble(line[2]));
            data.add(p);
        }

        Cart cart = new Cart(2){

            @Override
            protected MultivariateFunction createModel(Collection<PointValuePair> dataSet) {
                SummaryStatistics stats = new SummaryStatistics();
                for(PointValuePair p : dataSet){
                    stats.addValue(p.getSecond());
                }
                final double mean = stats.getMean();
                return new MultivariateFunction() {
                    @Override
                    public double value(double[] point) {
                        return mean;
                    }

                    @Override
                    public String toString() {
                        return String.format("%1.4s", mean);
                    }
                };
            }

            @Override
            public double errorFunction(Collection<PointValuePair> dataSet) {
                SummaryStatistics stats = new SummaryStatistics();
                for(PointValuePair p : dataSet){
                    stats.addValue(p.getSecond());
                }
                return stats.getVariance() * stats.getN();
            }

            @Override
            protected boolean shouldContinue(BestSplit split) {
                return split.bestError > 1.5 && split.left.size() > 1 && split.right.size() > 1;
            }
        };
        Cart.TreeNode result = cart.createTree(data);
        System.out.println("done");

    }

}