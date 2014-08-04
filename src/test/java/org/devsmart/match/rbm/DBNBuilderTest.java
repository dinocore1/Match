package org.devsmart.match.rbm;


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
        assertEquals(9, dbn.rbms.get(0).numVisible);
        assertEquals(3, dbn.rbms.get(0).numHidden);
        assertEquals(false, dbn.rbms.get(0).isGaussianVisible);
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
        assertEquals(8000, dbn.rbms.get(0).numVisible);
        assertEquals(500, dbn.rbms.get(0).numHidden);
        assertEquals(500, dbn.rbms.get(1).numVisible);
        assertEquals(300, dbn.rbms.get(1).numHidden);
        assertEquals(300, dbn.rbms.get(2).numVisible);
        assertEquals(2000, dbn.rbms.get(2).numHidden);
    }
}
