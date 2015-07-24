package org.devsmart.match.metric;

import org.junit.Test;
import static org.junit.Assert.*;

public class EvalTest {


    @Test
    public void testEval() {
        Evaluation<Boolean> eval = new Evaluation<Boolean>();

        String summary = eval.getSummaryTxt();

        eval.update(true, true);
        summary = eval.getSummaryTxt();
        assertEquals(1.0, eval.getAccuracy(), 0.00001);
        assertEquals(1, eval.getTruePositive());
        assertEquals(0, eval.getTrueNegitive());
        assertEquals(0, eval.getFalseNegitive());
        assertEquals(0, eval.getFalsePositives());

        eval.update(true, false);
        summary = eval.getSummaryTxt();
        assertEquals(0.5, eval.getAccuracy(), 0.00001);

        eval.update(false, false);
        summary = eval.getSummaryTxt();
        assertEquals(2, eval.getTruePositive());
        assertEquals(0.66666, eval.getAccuracy(), 0.001);

        eval.update(false, false);
        eval.update(false, false);
        eval.update(false, false);
        summary = eval.getSummaryTxt();



    }
}
