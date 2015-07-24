package org.devsmart.match.metric;

import org.junit.Test;
import static org.junit.Assert.*;

public class EvalTest {


    @Test
    public void testEval() {

        String[] gold = new String[] {"A", "B", "B", "B"};
        String[] pred = new String[] {"A", "A", "A", "A"};

        Evaluation<String> eval = new Evaluation<String>();

        for(int i=0;i<gold.length;i++){
            eval.update(gold[i], pred[i]);
        }

        System.out.println(eval.getSummaryTxt());

        assertEquals(1, eval.getTruePositive("A"));
        assertEquals(0, eval.getFalseNegitive("A"));
        assertEquals(3, eval.getFalseNegitive("B"));

        assertEquals(1d/4d, eval.getAccuracy(), 0.01);
        assertEquals(1d/4d, eval.getPrecision("A"), 0.01);
        assertEquals(1d/1d, eval.getRecall("A"), 0.01);
        assertEquals(4d/10d, eval.getF1Score(), 0.01);




    }
}
