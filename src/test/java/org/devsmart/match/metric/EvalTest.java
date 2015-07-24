package org.devsmart.match.metric;

import org.junit.Test;
import static org.junit.Assert.*;

public class EvalTest {


    @Test
    public void testEval() {
        Evaluation eval = new Evaluation();

        eval.update(1, 1);
        assertEquals(1.0, eval.getAccuracy(), 0.00001);
        assertEquals(1, eval.getTruePositive());
        assertEquals(0, eval.getTrueNegitive());
        assertEquals(0, eval.getFalseNegitive());
        assertEquals(0, eval.getFalsePositives());

        eval.update(1, 0);
        assertEquals(0.5, eval.getAccuracy(), 0.00001);

        eval.update(0, 0);
        assertEquals(2, eval.getTruePositive());
        assertEquals(0.66666, eval.getAccuracy(), 0.001);

    }
}
