package org.devsmart.match.metric;

import org.junit.Test;
import static org.junit.Assert.*;

public class EvalTest {

    @Test
    public void testF1() {

        Evaluation<String> eval = new Evaluation<String>();

        for(int i=0;i<95;i++){
            eval.update("A", "A");
        }

        for(int i=0;i<5;i++) {
            eval.update("B", "A");
        }

        assertEquals(0.95, eval.getAccuracy(), 0.001);
        assertEquals(0.655, eval.getF1Score(), 0.01);
    }

    @Test
    public void testConfusionMatrix() {
        String[] truth = new String[]     {"C", "A", "C", "C", "A", "B"};
        String[] predicted = new String[] {"A", "A", "C", "C", "A", "C"};

        ConfusionMatrix<String> confusionMatrix = new ConfusionMatrix<String>();
        for(int i=0;i<truth.length;i++) {
            confusionMatrix.update(truth[i], predicted[i]);
        }


        assertEquals(2, confusionMatrix.getCount("A", "A"));
        assertEquals(0, confusionMatrix.getCount("A", "B"));
        assertEquals(0, confusionMatrix.getCount("A", "C"));

        assertEquals(0, confusionMatrix.getCount("B", "A"));
        assertEquals(0, confusionMatrix.getCount("B", "B"));
        assertEquals(1, confusionMatrix.getCount("B", "C"));

        assertEquals(1, confusionMatrix.getCount("C", "A"));
        assertEquals(0, confusionMatrix.getCount("C", "B"));
        assertEquals(2, confusionMatrix.getCount("C", "C"));

        assertEquals(2, confusionMatrix.getTruePositive("A"));
        assertEquals(0, confusionMatrix.getTruePositive("B"));
        assertEquals(2, confusionMatrix.getTruePositive("C"));

        assertEquals(0, confusionMatrix.getFalsePositive("A"));
        assertEquals(1, confusionMatrix.getFalsePositive("B"));
        assertEquals(1, confusionMatrix.getFalsePositive("C"));

        assertEquals(2, confusionMatrix.getTrueNegitive("A"));
        assertEquals(4, confusionMatrix.getTrueNegitive("B"));
        assertEquals(2, confusionMatrix.getTrueNegitive("C"));

        assertEquals(1, confusionMatrix.getFalseNegitive("A"));
        assertEquals(0, confusionMatrix.getFalseNegitive("B"));
        assertEquals(1, confusionMatrix.getFalseNegitive("C"));

    }


    @Test
    public void testEval() {

        String[] gold = new String[] {"A", "B", "B", "B"};
        String[] pred = new String[] {"A", "A", "A", "A"};

        Evaluation<String> eval = new Evaluation<String>();

        for(int i=0;i<gold.length;i++){
            eval.update(gold[i], pred[i]);
        }

        System.out.println(eval.getSummaryTxt());

        assertEquals(4d/10d, eval.getFScore(1.0, "A"), 0.001);

        assertEquals(1d / 4d, eval.getAccuracy(), 0.01);
        assertEquals(1d / 2d, eval.getPrecision(), 0.01);
        assertEquals(1d / 4d, eval.getRecall(), 0.01);
        assertEquals(1d / 3d, eval.getF1Score(), 0.01);

    }

    @Test
    public void testThing() {
        String[] truth = new String[]     {"C", "A", "C", "C", "A", "B"};
        String[] predicted = new String[] {"A", "A", "C", "C", "A", "C"};

        Evaluation<String> eval = new Evaluation<String>();

        for(int i=0;i<truth.length;i++) {
            eval.update(truth[i], predicted[i]);
        }

        System.out.println(eval.getSummaryTxt());
    }
}
