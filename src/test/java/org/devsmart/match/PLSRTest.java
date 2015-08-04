package org.devsmart.match;


import au.com.bytecode.opencsv.CSVReader;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.commons.math3.util.ResizableDoubleArray;
import org.junit.Test;
import static org.junit.Assert.*;

import java.io.FileReader;
import java.util.ArrayList;

public class PLSRTest {

    private RealMatrix toMatrix(ArrayList<double[]> input) {
        double[] row = input.get(0);

        RealMatrix X = MatrixUtils.createRealMatrix(input.size(), row.length);
        for(int i=0;i<input.size();i++) {
            row = input.get(i);
            X.setRow(i, row);
        }
        return X;
    }

    @Test
    public void testAspenDataSet() throws Exception {
        CSVReader reader = new CSVReader(new FileReader("data/plsDatast.csv"));

        //read header
        String[] line = reader.readNext();

        ArrayList<double[]> xvalues = new ArrayList<double[]>();
        ResizableDoubleArray yvalues = new ResizableDoubleArray();

        while((line = reader.readNext()) != null) {

            yvalues.addElement(Double.parseDouble(line[1]));

            double[] x = new double[line.length - 2];
            for(int i=2;i<line.length;i++) {
                x[i-2] = Double.parseDouble(line[i]);
            }
            xvalues.add(x);
        }

        PLSR plsr = new PLSR(toMatrix(xvalues), yvalues.getElements(), 2);

        SimpleRegression regression = new SimpleRegression(true);

        for(int i=0;i<xvalues.size();i++){
            double y = plsr.value(xvalues.get(i));
            regression.addData(yvalues.getElement(i), y);
            System.out.println(String.format("i:%d y = %f", i, y));
        }

        final double r2 = regression.getRSquare();
        System.out.println(String.format("r2 = %f", r2));

        assertTrue(r2 > 0.99);

    }
}
