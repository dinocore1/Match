package org.devsmart.match.rbm;

import java.awt.image.BufferedImage;
import java.io.*;

public class MNISTImageFile {

    private final RandomAccessFile mRandomAccessFile;
    public final int numImages;
    public final int height;
    public final int width;

    public MNISTImageFile(File file) throws IOException {
        mRandomAccessFile = new RandomAccessFile(file, "r");
        mRandomAccessFile.seek(0);

        if(readInt32() != 0x803){
            throw new IOException("wrong magic number");
        }

        numImages = readInt32();
        height = readInt32();
        width = readInt32();

    }

    private int readInt32() throws IOException {
        byte[] data = new byte[4];
        mRandomAccessFile.readFully(data);
        int retval = (0xff000000 & data[0] << 24) + (0xff0000 & data[1] << 16) + (0xff00 & data[2] << 8) + (0xff & data[3]);
        return retval;
    }

    public double[] getImage(int index) throws IOException {
        long offset = index* width * height +16;
        mRandomAccessFile.seek(offset);

        byte[] data = new byte[width * height];
        mRandomAccessFile.readFully(data);

        double[] retval = new double[data.length];
        for(int i=0;i<retval.length;i++){
            retval[i] = (0xff & data[i]) / 255.0;
        }
        return retval;
    }

    public BufferedImage createImage(double[] data) {
        BufferedImage retval = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        int i=0;
        for(int r=0;r< height;r++){
            for(int c=0;c< width;c++){
                int color = 0xff & (int)(255 * data[i++]);
                //retval.setRGB(c, r, new Color(color,color,color).getRGB());
                retval.setRGB(c, r, (0xff << 24) | ((0xff & color) << 16) | ((0xff & color) << 8) | (color & 0xff));
            }
        }
        return retval;
    }

    public BufferedImage getBufferedImage(int index) throws IOException {
        long offset = index* width * height +16;
        mRandomAccessFile.seek(offset);
        BufferedImage retval = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        byte[] data = new byte[width * height];
        mRandomAccessFile.readFully(data);
        int i=0;
        for(int r=0;r< height;r++){
            for(int c=0;c< width;c++){
                int color = 0xff & data[i++];
                //retval.setRGB(c, r, new Color(color,color,color).getRGB());
                retval.setRGB(c, r, (0xff << 24) | ((0xff & color) << 16) | ((0xff & color) << 8) | (color & 0xff));
            }
        }
        return retval;

    }




}
