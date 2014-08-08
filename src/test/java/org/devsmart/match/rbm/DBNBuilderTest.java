package org.devsmart.match.rbm;


import org.devsmart.match.rbm.nuron.BernoulliNuron;
import org.junit.Test;
import static org.junit.Assert.*;

public class DBNBuilderTest {


    @Test
    public void testBuildDBN1() {
        DBNBuilder builder = new DBNBuilder();
        builder.addLayer(9, false);
        builder.addLayer(3, false);

        DBN dbn = builder.build();
        assertNotNull(dbn);
        assertEquals(1, dbn.rbms.size());
        assertEquals(9, dbn.rbms.get(0).visible.length);
        assertEquals(3, dbn.rbms.get(0).hidden.length);
        assertTrue(dbn.rbms.get(0).visible[0] instanceof BernoulliNuron);
    }

    @Test
    public void testBuildDBN2() {
        DBNBuilder builder = new DBNBuilder();
        builder.addLayer(8000, true);
        builder.addLayer(500, false);
        builder.addLayer(300, false);
        builder.addLayer(2000, false);

        DBN dbn = builder.build();
        assertEquals(3, dbn.rbms.size());
        assertEquals(8000, dbn.rbms.get(0).visible.length);
        assertEquals(500, dbn.rbms.get(0).hidden.length);
        assertEquals(500, dbn.rbms.get(1).visible.length);
        assertEquals(300, dbn.rbms.get(1).hidden.length);
        assertEquals(300, dbn.rbms.get(2).visible.length);
        assertEquals(2000, dbn.rbms.get(2).hidden.length);
    }
}
