package org.devsmart.match.metric;

import org.junit.Test;
import static org.junit.Assert.*;

public class EvalTest {


    @Test
    public void testEval() {

        int[] gold = new int[] {1, 0, 0, 0};
        int[] pred = new int[] {1, 1, 1, 1};

        Evaluation<Integer> eval = new Evaluation<Integer>();

        for(int i=0;i<gold.length;i++){
            eval.update(gold[i], pred[i]);
        }

        System.out.println(eval.getSummaryTxt());

        assertEquals(1, eval.getTruePositive(1));

        assertEquals(3, eval.getFalseNegitive(0));

        assertEquals(1d/4d, eval.getAccuracy(), 0.01);
        assertEquals(1d/4d, eval.getPrecision(1), 0.01);
        assertEquals(1d/1d, eval.getRecall(1), 0.01);
        assertEquals(4d/10d, eval.getF1Score(), 0.01);




    }
}
