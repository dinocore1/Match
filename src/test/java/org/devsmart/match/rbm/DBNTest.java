package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class DBNTest {

    private RealVector createDigit(int digit){
        ArrayRealVector retval = null;

        switch (digit){
            case 0:
                retval = new ArrayRealVector(new double[]{
                        1, 1, 1,
                        1, 0, 1,
                        1, 0, 1,
                        1, 0, 1,
                        1, 1, 1
                });
                break;
            case 1:
                retval = new ArrayRealVector(new double[]{
                        0, 0, 1,
                        0, 0, 1,
                        0, 0, 1,
                        0, 0, 1,
                        0, 0, 1
                });
                break;
            case 2:
                retval = new ArrayRealVector(new double[]{
                        1, 1, 1,
                        0, 0, 1,
                        1, 1, 1,
                        1, 0, 0,
                        1, 1, 1
                });
                break;
            case 3:
                retval = new ArrayRealVector(new double[]{
                        1, 1, 1,
                        0, 0, 1,
                        1, 1, 1,
                        0, 0, 1,
                        1, 1, 1
                });
                break;
            case 4:
                retval = new ArrayRealVector(new double[]{
                        1, 0, 1,
                        1, 0, 1,
                        1, 1, 1,
                        0, 0, 1,
                        0, 0, 1
                });
                break;
            case 5:
                retval = new ArrayRealVector(new double[]{
                        1, 1, 1,
                        1, 0, 0,
                        1, 1, 1,
                        0, 0, 1,
                        1, 1, 1
                });
                break;
            case 6:
                retval = new ArrayRealVector(new double[]{
                        1, 0, 0,
                        1, 0, 0,
                        1, 1, 1,
                        1, 0, 1,
                        1, 1, 1
                });
                break;
            case 7:
                retval = new ArrayRealVector(new double[]{
                        1, 1, 1,
                        0, 0, 1,
                        0, 0, 1,
                        0, 0, 1,
                        0, 0, 1
                });
                break;
            case 8:
                retval = new ArrayRealVector(new double[]{
                        1, 1, 1,
                        1, 0, 1,
                        1, 1, 1,
                        1, 0, 1,
                        1, 1, 1
                });
                break;
            case 9:
                retval = new ArrayRealVector(new double[]{
                        1, 1, 1,
                        1, 0, 1,
                        1, 1, 1,
                        0, 0, 1,
                        0, 0, 1
                });
                break;
        }

        return retval;
    }

    @Test
    public void testTrainDBN() {
        DBN dbn = new DBNBuilder().addLayer(15, false)
                .addLayer(10, false)
                .addLayer(10, false)
                .withRandom(new Random(1))
                .build();

/*
        DBNTrainer trainer = new DBNTrainer(dbn);
        ArrayList<RealVector> trainingData = new ArrayList<RealVector>();
        for(int i=0;i<10;i++){
            trainingData.add(createDigit(i));
        }
        trainer.train(trainingData, 100000);

        for(int i=0;i<10;i++) {
            RealVector hidden = dbn.propagateUp(createDigit(i), 2);
            System.out.println(String.format("Digit: %d: %s", i, hidden));
        }

        for(int i=0;i<10;i++){
            ArrayRealVector hidden = new ArrayRealVector(10);
            hidden.setEntry(i, 1.0);
            RealVector visible = dbn.propagateDown(hidden, 2);

            System.out.println(String.format("\n%d: \n%s\n%s\n%s\n%s\n%s", i,
                    visible.getSubVector(0, 3),
                    visible.getSubVector(3, 3),
                    visible.getSubVector(6, 3),
                    visible.getSubVector(9, 3),
                    visible.getSubVector(12, 3)
                    ));
        }

*/
    }
}
